"""Read-only daily production figures shared by planning and dashboard."""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


def daily_production_totals(fetch_one, today=None):
    tz = ZoneInfo("America/Guatemala")
    today = today or datetime.now(tz).date()
    start = datetime.combine(today, time.min, tzinfo=tz)
    lower = start.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    upper = (start + timedelta(days=1)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    goal = fetch_one(
        "SELECT target_garments,target_batches FROM daily_production_goals WHERE goal_date=%s LIMIT 1",
        (today,),
    ) or {}
    planned = fetch_one("""
        SELECT COUNT(*) planned_batches,
            COALESCE(SUM(b.planned_quantity),0) planned_garments,
            COALESCE(SUM(GREATEST(b.planned_quantity-COALESCE(i.n,0),0)),0) pending_inspection,
            COALESCE(SUM(COALESCE(i.n,0)<b.planned_quantity),0) pending_batches,
            COALESCE(SUM(b.status='REVISION_PENDIENTE'),0) pending_closure
        FROM batches b
        LEFT JOIN (SELECT batch_id,COUNT(*) n FROM inspections GROUP BY batch_id) i ON i.batch_id=b.id
        WHERE b.production_date=%s AND b.active=1
        """, (today,)) or {}
    # Los lotes eliminados lógicamente siguen con sus inspecciones intactas,
    # pero sus prendas dejan de contar en el desempeño del día.
    actual = fetch_one("""
        SELECT COUNT(*) inspected_today,
            COALESCE(SUM(ai_decision='ANOMALIA' AND COALESCE(review_status,'PENDIENTE')='PENDIENTE'),0) pending_alerts
        FROM inspections i
        LEFT JOIN batches b ON b.id=i.batch_id
        WHERE i.created_at >= %s AND i.created_at < %s
          AND COALESCE(b.active,1)=1
        """, (lower, upper)) or {}
    result = {key: int(value or 0) for row in (goal, planned, actual) for key, value in row.items()}
    result['target_garments'] = int(goal.get('target_garments') or 0)
    result['target_batches'] = int(goal.get('target_batches') or 0)
    result['remaining_to_goal'] = max(result['target_garments'] - result.get('inspected_today', 0), 0)
    result['remaining_to_plan'] = max(result['target_garments'] - result.get('planned_garments', 0), 0)
    return result
