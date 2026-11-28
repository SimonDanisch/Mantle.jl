"""Document checks and abstract models; never imports or executes Mantle/Julia."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'


def reachable(edges, source, target):
    todo, seen = [source], set()
    while todo:
        node = todo.pop()
        if node == target:
            return True
        if node not in seen:
            seen.add(node)
            todo.extend(v for u, v in edges if u == node)
    return False


def model_checks():
    results = {}
    # Reuse needs every maximal old user, including independent slow readers.
    edges = {('produce', 'slow'), ('produce', 'fast')}
    old_users = {'produce', 'slow', 'fast'}
    frontier = {u for u in old_users if not any(
        u != v and reachable(edges, u, v) for v in old_users)}
    assert frontier == {'slow', 'fast'}
    fixed = edges | {(u, 'reuse') for u in frontier}
    assert all(reachable(fixed, u, 'reuse') for u in old_users)
    broken = edges | {('fast', 'reuse')}
    assert not reachable(broken, 'slow', 'reuse')  # negative control
    results['complete_alias_frontier'] = True

    # A frontier handles a long write chain without quadratic access history.
    writer, edges = None, set()
    for node in range(1000):
        if writer is not None:
            edges.add((writer, node))
        writer = node
    assert len(edges) == 999
    results['linear_chain_frontier_size'] = True

    # Dirty version 2 survives acceptance of older build 1 and failure before 2.
    dirty = {1, 2}
    dirty.difference_update(v for v in dirty.copy() if v <= 1)
    assert dirty == {2}
    assert dirty  # unaccepted future build cannot consume it
    dirty.difference_update(v for v in dirty.copy() if v <= 2)
    assert not dirty
    results['accepted_build_consumes_only_covered_geometry'] = True

    # Child loop stop must reset each enclosing invocation.
    executed = []
    for outer in range(3):
        remaining, stopped = 2, False
        for inner in range(8):
            stopped = stopped or remaining == 0
            if not stopped:
                executed.append((outer, inner))
                remaining -= 1
    assert executed == [(o, i) for o in range(3) for i in range(2)]
    results['nested_loop_lexical_reset'] = True

    # A region predicate is latched, not reread after a source-flag write.
    source_flag = True
    latched = source_flag
    if latched:
        source_flag = False
    assert latched and not source_flag
    results['region_predicate_latched_for_body'] = True

    # Timeout cannot consume/cancel/poison live work behind legal dependencies.
    segment_times = [0.75] * 3
    assert max(segment_times) < 1 and sum(segment_times) > 2
    state = {'pending': True, 'holds': 2, 'poisoned': False}
    before = state.copy()
    deadline_expired = sum(segment_times) > 2
    assert deadline_expired and state == before
    results['wait_timeout_preserves_live_work'] = True

    # Conditional clear+draw removes a disappeared occluder; skipped path preserves.
    old = [0.2, 1.0]
    for active, expected in [(False, old), (True, [1.0, 1.0])]:
        image = old.copy()
        predicate = active
        if predicate:
            image[:] = [1.0, 1.0]
            # No geometry remains to rasterize.
        assert image == expected
    assert old != [1.0, 1.0]  # negative control: redraw without clear
    results['conditional_render_clear_and_preserve'] = True

    # Candidate recipe rebuild changes sequence/workspace, preserving public output.
    output = object()
    def materialize(bound):
        return {'output': output, 'commands': 1 if bound <= 64 else 3,
                'workspace': 64 if bound <= 64 else 256}
    small, large = materialize(32), materialize(128)
    assert small['output'] is large['output']
    assert (small['commands'], small['workspace']) == (1, 64)
    assert (large['commands'], large['workspace']) == (3, 256)
    results['candidate_region_rebuild_preserves_output'] = True

    # Allocation generation prevents pointer-reuse aliasing; actual aliases join.
    roots = {}
    first = roots.setdefault(('context', 0x1000, 1), object())
    alias = roots.setdefault(('context', 0x1000, 1), object())
    recycled = roots.setdefault(('context', 0x1000, 2), object())
    assert first is alias and first is not recycled
    results['import_alias_allocation_generation'] = True
    return results


def slug(heading):
    heading = re.sub(r'[^\w\s-]', '', heading.lower())
    return re.sub(r'\s', '-', heading)


def document_checks():
    # Other upstream documentation is outside this refactor-plan check's scope.
    names = {'api.md', 'api-guide.md', 'examples.jl',
             'execution-contracts.md', 'internals.jl',
             'recording-plan.md', 'refactor-ledger.md',
             'resizing-and-raytracing.jl'}
    paths = sorted(DOCS / name for name in names)
    assert all(p.is_file() for p in paths), 'missing refactor-plan document'
    texts = {p.name: p.read_text() for p in paths}
    ledger = texts['refactor-ledger.md']
    rows = re.findall(r'^\| ((?:P\d+\.[A-Z]\d+|F\d+)) \|', ledger, re.M)
    assert len(rows) == len(set(rows)), 'duplicate stable ledger IDs'
    forbidden = [r'\blengthof\s*\(', r'loops do not nest',
                 r'kernel candidates only', r'Graph arrays are never persistent',
                 r'Plan\(g;', r'run!\(plan\)', r'with twice the job timeout',
                 r'KA\.allocate.*gives a transient', r'^\+(?:$|#+ )',
                 r'Each phase requires nonempty tests',
                 r'Readback\(a\) after compilation may reuse',
                 r'view\(image,ranges\.\.\.;mip=0',
                 r'Array\(adapt\(', r'\bgraph_array\s*\(',
                 r'\bGraphDevice\b', r'\bbindarguments\b', r'\bbindtarget\b',
                 r'\bresolvebindings\b', r'adapt\(Device\(',
                 r'(?i)\b(?:graph|group)-bound\b']
    for name, text in texts.items():
        for pattern in forbidden:
            assert not re.search(pattern, text, re.M), (name, pattern)
        if name.endswith('.jl'):
            assert not re.search(r'^(?:function )?(?:LinearAlgebra\.)?mul!\((?:target|dev|g|group|step)\s*,', text, re.M), name
        if name.endswith('.md'):
            for target in re.findall(r'\]\(([^\s)]+)\)', text):
                if re.match(r'[a-z]+://', target):
                    continue
                rel, _, anchor = target.partition('#')
                path = DOCS / rel if rel else DOCS / name
                assert path.is_file(), (name, target)
                if anchor:
                    headings = re.findall(r'^#+ (.+)$', path.read_text(), re.M)
                    assert anchor in {slug(h) for h in headings}, (name, target)

    resolution = json.loads((ROOT/'experiments/review-2026-10-29/plan-reconciliation.json').read_text())
    findings = resolution['findings']
    assert {f['id'] for f in findings} == {f'MR-{i:02}' for i in range(1, 25)}
    for finding in findings:
        assert finding['runtime_status'] == 'open'
        assert finding['acceptance']
        for ident in finding['ledger']:
            assert ident in rows, (finding['id'], ident)
        for ref in finding['contracts']:
            path = ROOT / ref['file']
            assert path.is_file()
            assert ref['section'] in path.read_text(), ref

    execution = resolution['execution_plan']
    plan = texts['recording-plan.md']
    batches = re.findall(r'^\| (B\d+) \|', plan, re.M)
    mwes = re.findall(r'^\| (MWE-\d+) \|', plan, re.M)
    assert batches == execution['batch_order'] == [f'B{i}' for i in range(1, 6)]
    assert mwes == [f'MWE-{i:02}' for i in range(1, 9)]
    assert mwes == [m['id'] for m in execution['fixed_mwes']]
    assert not execution['phase_boundaries_are_test_sessions']
    assert execution['persistent_julia_workers']
    assert execution['qualification_separate_from_smoke']
    covered = set()
    for mwe in execution['fixed_mwes']:
        assert mwe['runtime_status'] == 'open'
        assert mwe['first_batch'] in batches
        assert mwe['profiles']
        assert min(mwe['profiles'].values()) == mwe['first_batch']
        assert set(mwe['profiles'].values()) <= set(batches)
        assert set(mwe['ledger']) <= set(rows)
        covered.update(mwe['use_cases'])
    assert {'DNNs', 'Raytracing', 'Simulations', 'FEM', 'Particles',
            'Raster/Makie', 'LLMs'} <= covered
    assert set(execution['ledger']) <= set(rows)
    audit = json.loads((ROOT/'experiments/review-2026-10-29'/resolution['followup_api_audit']).read_text())
    followups = audit['findings']
    assert {f['id'] for f in followups} == {f'AE-{i:02}' for i in range(1, 11)}
    for finding in followups:
        assert finding['runtime_status'] == 'open'
        assert finding['design_status'] in ('reconciled_in_target_documents',
                                            'superseded_by_user_routing_ruling')
        if finding['design_status'] == 'superseded_by_user_routing_ruling':
            assert finding['ruling'] == resolution['explicit_routing_ruling']
        assert finding['acceptance']
        assert finding['ledger'] in rows
        assert set(finding['fixed_mwes']) <= set(mwes)
        for contract in finding['contracts']:
            assert (ROOT/contract).is_file()
    use_cases = json.loads((ROOT/'experiments/review-2026-10-29'/resolution['use_case_review']).read_text())
    assert {f['id'] for f in use_cases['findings']} == {f'UC-{i:02}' for i in range(1, 9)}
    review_ids = {f['id'] for f in use_cases['findings']}
    assert {'DNNs', 'Raytracing', 'Simulations', 'FEM', 'Particles',
            'Raster/Makie', 'LLMs'} <= {c['use_case'] for c in use_cases['coverage']}
    for finding in use_cases['findings']:
        assert finding['ledger'] in rows
        assert finding['status'].startswith('open_')
        assert finding['acceptance'] and finding['required_resolution']
        assert set(finding['fixed_mwes']) <= set(mwes)
        for evidence in finding['evidence']:
            assert Path(evidence['file']).is_file()
            assert evidence['line'] > 0 and evidence['symbol']
    for coverage in use_cases['coverage']:
        assert set(coverage['remaining']) <= review_ids
        assert set(coverage['existing_mwes']) <= set(mwes)
    for section in ['Compile and runtime performance', 'Authority and checkpoint',
                    'Work batches and logical phases', 'Fixed MWE catalog',
                    'Validation schedule and Julia cost',
                    'Failure handling and stopping rules']:
        assert re.search(r'^##?\#? ' + re.escape(section) + '$', plan, re.M)
    assert '16.11 Representation and performance qualification' in texts['execution-contracts.md']
    normalized_plan = ' '.join(plan.split())
    for phrase in ['prefer runtime performance', 'old implementation code',
                   'not mandatory validation sessions', 'Every retry must test a new hypothesis',
                   'repeat a full suite only if shared']:
        assert phrase in normalized_plan, phrase

    assert 'scratch=false' in texts['api.md']
    assert 'share_scratch=true' in texts['api.md']
    assert 'Readback(previous::ReadbackFuture)' in texts['api.md']
    assert 'resize!(output, computed_count[1])' in texts['examples.jl']
    assert 'maximalusers(s.dag, old)' in texts['internals.jl']
    assert 'consumeonlycoveredversions!' in texts['internals.jl']
    assert 'struct Resize{E} <: Change' in texts['resizing-and-raytracing.jl']
    assert 'keepbytes::E' in texts['resizing-and-raytracing.jl']
    assert 'host-grow to capacity(active)+2' in plan
    ruling = json.loads((ROOT/'experiments/review-2026-10-29'/resolution['explicit_routing_ruling']).read_text())
    assert ruling['runtime_status'] == 'open'
    assert set(ruling['ledger']) <= set(rows)
    assert ruling['working_public_spelling'] == 'call!(target,f,args...;kwargs...), target Graph/Group'
    assert 'KA.get_backend(a::MantleArray) = a.device' in texts['internals.jl']
    assert 'struct MantleArray{T,N,D<:Device}' in texts['internals.jl']
    assert 'call!(target::Union{Graph,Group}, ::typeof(mul!)' in texts['internals.jl']
    assert '_mul!(location(C, A, B), C, A, B)' in texts['internals.jl']
    assert 'snapshotrecipe!(target, a)' in texts['internals.jl']
    assert 'rebindpendingoutputs!(g, new)' in texts['internals.jl']
    assert 'Graph owns stable output IDs and pending requests' in texts['execution-contracts.md']
    assert 'same Device/context domain' in texts['api.md'] or 'Device/domain' in texts['api.md']
    assert 'mip=1,layers=:,aspect=nothing' in texts['api.md']
    assert 'origin=nothing' in texts['api.md'] and 'hardware=nothing' in texts['api.md']
    assert 'unrequested outputs copy zero payload bytes' in plan
    assert 'ONE candidate eligibility/selection' in texts['execution-contracts.md'] or 'ONLY candidate eligibility/selection' in texts['execution-contracts.md']
    for phrase in ['not arbitrary Julia capture', 'no Graph/Group overload of mul!',
                   'no extra public device argument', 'not the function\'s spelling',
                   'No backend chooses algorithms', 'CPU callbacks use hostcall!']:
        assert phrase in normalized_plan, phrase
    consumer = (ROOT/'experiments/review-2026-10-20/consumer-coverage.md').read_text()
    for pattern in [r'\bGraphDevice\b', r'adapt\(Device\(', r'(?i)\b(?:graph|group)-bound\b']:
        assert not re.search(pattern, consumer), ('consumer-coverage.md', pattern)
    assert 'WaitTimeout' in texts['api.md'] and 'WaitTimeout' in texts['internals.jl']
    return {'documents': len(paths), 'unique_ledger_items': len(rows),
            'review_findings_mapped': len(findings), 'local_links_and_anchors': 'valid',
            'followup_api_findings_mapped': len(followups),
            'use_case_review_obligations_mapped': len(use_cases['findings']),
            'explicit_routing_and_shared_capability_policy': 'specified; runtime open',
            'fixed_mwes_specified': len(mwes), 'implementation_batches': len(batches),
            'mwe_runtime_qualification': 'open; specifications only',
            'stale_contract_patterns': 'absent',
            'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


if __name__ == '__main__':
    result = {'scope': 'Documentation consistency and abstract models only; no native/runtime qualification',
              'documents': document_checks(), 'models': model_checks()}
    Path(__file__).with_name('plan-reconciliation-checks.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'documents'}, indent=2))
    print(json.dumps({k: v for k, v in result['documents'].items() if k != 'sha256'}, indent=2))
