"""History filter contract using its real handler/template and read-only SQL doubles."""
import ast
from pathlib import Path
import unittest

from flask import Flask, request, session, render_template, flash, redirect, url_for
from jinja2 import ChoiceLoader, DictLoader, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]


def create_fixture_app():
    app = Flask(__name__, static_folder=str(ROOT / 'static'))
    app.secret_key = 'local-history-filter-fixture'
    app.jinja_loader = ChoiceLoader([
        DictLoader({'base.html': '<!doctype html><html lang="es"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Historial</title><body>{% block content %}{% endblock %}</body></html>'}),
        FileSystemLoader(str(ROOT / 'templates')),
    ])
    models = [dict(id=i, code=f'BL-{i:04}', name=f'Modelo {i}', active=True, status='APROBADO') for i in range(1, 6)]
    models += [dict(id=990, code='HIST-990', name='Blusa Ámbar', active=False),
               dict(id=991, code='ACT-991', name='Blusa Ámbar', active=True),
               dict(id=992, code='SAFE-992', name='<img src=x onerror=alert(1)>', active=False)]
    for model in models[5:]:
        model.update(status='APROBADO', active=False)
    models += [dict(id=993, code='TEST-AI-993', name='Prueba antigua', active=False, status='APROBADO'),
               dict(id=994, code='DRAFT-994', name='Borrador', active=False, status='BORRADOR')]
    calls = []

    def fetch_all(sql, params=()):
        calls.append((sql, params))
        compact = ' '.join(sql.split())
        if compact.startswith('SELECT gm.id, gm.code, gm.name') and 'FROM garment_models gm' in compact:
            assert 'WHERE gm.active = 1' in compact
            approved_only="gm.status = 'APROBADO'" in compact
            return [m for m in models if m['active'] and (not approved_only or m['status']=='APROBADO')]
        if compact.startswith('SELECT id, code FROM batches'):
            return [dict(id=121, code='LOT-121')]
        if 'FROM users' in compact:
            return [dict(id=41, name='Responsable')]
        return []

    def fetch_one(sql, params=()):
        calls.append((sql, params))
        return dict(inspected=0, passed=0, rejected=0, alerts=0, discarded=0, pending=0)

    tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'garment_catalog_where']
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == 'records')
    node.decorator_list = []
    from types import SimpleNamespace
    env = dict(request=request, session=session, normalize_role=lambda role:role or 'ADMIN',ROLE_QUALITY_MANAGER='QUALITY', render_template=render_template, flash=flash,
               redirect=redirect, url_for=url_for, fetch_all=fetch_all,
               fetch_one=fetch_one, get_garment_presentation_urls=lambda ids: {},
               checkpoint_is_available=lambda *a:True,get_ai_artifacts_root=lambda:'.',
               ai_validation=SimpleNamespace(is_technically_invalidated=lambda notes:False))
    exec(compile(ast.Module(body=[*helpers,node], type_ignores=[]), str(ROOT / 'app.py'), 'exec'), env)
    app.add_url_rule('/registros', 'records', env['records'])
    app.fixture_calls = calls
    app.fixture_models = models
    return app


class HistoryModelFilterTests(unittest.TestCase):
    def setUp(self):
        self.app = create_fixture_app()
        self.client = self.app.test_client()

    def test_five_current_catalog_models_include_ai_preparation_pending(self):
        response = self.client.get('/registros?model_id=990')
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        for excluded in ['HIST-990','ACT-991','TEST-AI-993','DRAFT-994']:
            self.assertNotIn(excluded, text)
        import re
        selector=re.search(r'<select id="recordsModel".*?</select>',text,re.S).group()
        self.assertEqual(selector.count('<option'),6)
        self.assertIn('BL-0005',selector)
        self.assertIn('value="">Todos', text)

    def test_catalog_changes_are_dynamic_and_names_are_escaped(self):
        self.app.fixture_models.append(dict(id=995,code='NEW-995',name='<img src=x onerror=alert(1)>',active=True,status='APROBADO'))
        text=self.client.get('/registros?model_id=995').get_data(as_text=True)
        self.assertIn('NEW-995',text)
        self.assertIn('&lt;img src=x onerror=alert(1)&gt;',text)
        self.assertNotIn('<img src=x onerror=alert(1)>',text)

    def test_consultar_keeps_existing_parameterized_model_and_other_filters(self):
        response = self.client.get('/registros?q=costura&date_from=2026-10-01&date_to=2026-10-22&batch_id=121&model_id=990&reviewer_id=41&result=APTAS')
        self.assertEqual(response.status_code, 200)
        filtered = [(sql, params) for sql, params in self.app.fixture_calls if 'b.garment_model_id = %s' in sql]
        self.assertTrue(filtered)
        for sql, params in filtered:
            self.assertIn(990, params)
            self.assertIn(121, params)
            self.assertIn(41, params)
            self.assertIn('%costura%', params)
        self.assertTrue(all(sql.lstrip().upper().startswith('SELECT') for sql, _ in self.app.fixture_calls))

    def test_todos_does_not_apply_a_model_condition(self):
        self.assertEqual(self.client.get('/registros?model_id=').status_code, 200)
        self.assertFalse(any('b.garment_model_id = %s' in sql for sql, _ in self.app.fixture_calls))


if __name__ == '__main__':
    unittest.main()
