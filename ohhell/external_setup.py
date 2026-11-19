"""Pinned, local-only acquisition of upstream Oh Hell-family bot assets.

Upstream files stay under gitignored runs/. They are never silently replaced
by home-grown players. Source, patches and model hashes accompany league runs.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'runs' / 'external_bots'
REVISIONS = {
    'campbell': ('campbellsoup37/OhHell', '2f4d3dd3600d1df1cf439258b6150aee3f50ce7d'),
    'diminishing': ('thijssnelleman/DiminishingBridge', '305af70fe0c5341aeb70d5e9d2411bb57c4e7c67'),
    'drl': ('jonas-schumacher/drl-pomdp', '1acb47b18959176bc654f10b9710209307ad8098'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths():
    yield 'diminishing', 'main.cc'
    yield 'diminishing', 'README.md'
    yield 'drl', 'agent.py'
    yield 'drl', 'README.md'
    yield 'drl', 'environment.py'
    for path in ('README.md', 'OhHell/src/strategyOI/AiStrategyModuleOI.java',
                 'OhHell/src/strategyOI/OverallValueLearner.java',
                 'OhHell/src/strategyOI/ImmediateValueLearner.java',
                 'OhHell/src/core/Deck.java', 'OhHell/src/ml/SparseVector.java',
                 'OhHell/src/ml/Layer.java'):
        yield 'campbell', path
    for n in range(2, 9):
        for model in ('ovl', 'ivl'):
            yield 'campbell', f'OhHell/resources/models/N{n}/D1/T0/{model}.txt'


def acquire():
    previous = ASSETS / 'manifest.json'
    old = json.loads(previous.read_text()) if previous.exists() else {}
    if old and old.get('revisions') != {k: list(v) for k, v in REVISIONS.items()}:
        raise ValueError('External assets use different revisions; preserve that folder and choose a fresh checkout')

    def download(item):
        name, path = item
        target = ASSETS / name / path
        relative = str(target.relative_to(ASSETS).as_posix())
        expected = old.get('files', {}).get(relative)
        if target.exists():
            if not expected or sha256(target) != expected:
                raise ValueError(f'Unverified or modified upstream asset: {target}')
            return relative, expected
        repo, commit = REVISIONS[name]
        request = Request(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}',
                          headers={'User-Agent': 'OhHell-local-research-league'})
        with urlopen(request, timeout=45) as response:
            data = response.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return relative, sha256(target)

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(download, source_paths()))
    manifest = {'revisions': {k: list(v) for k, v in REVISIONS.items()}, 'files': files,
                'scope': 'Local research copies; upstream licence/attribution retained. '
                'No licence found in the pinned Oh Hell-family repository roots; not vendored for redistribution.'}
    previous.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def patched_diminishing(source):
    """Compatibility/safety patches; the upstream MC and MCTS algorithms remain.

    Fifteen rank slots encode bower power. Nonexistent slots are marked known
    absent. All actual legality/ranking enters through the Python card tables.
    """
    replacements = [
        ('#define MAX_ROUNDS 13', '#define MAX_ROUNDS 15'),
        ('#define MAX_PLAYERS 4', '#define MAX_PLAYERS 8'),
        ('#define WIN_FACT 1 ', '#define WIN_FACT 10 '),
        ('#define LOSE_FACT 1', '#define LOSE_FACT 10'),
        ('int main(int argc, char *argv[])', 'int upstream_main(int argc, char *argv[])'),
        # Original stack root was passed to a function that calls delete.
        ('deleteMCTSTree(rootPointer);',
         'for (int k = 0; k < root.numberOfChildren; k++) deleteMCTSTree(root.children[k]);'),
        ('while(node->children[child] != NULL && child < (MAX_CARDS - MAX_ROUNDS))',
         'while(child < MAX_CARDS && node->children[child] != NULL)'),
        ('delete currentNode;\n\t\t\t\tcurrentNode = currentNode->parent;',
         'delete currentNode;\n\t\t\t\tcurrentNode = temp;'),
        # After pruning upward, childNumber describes the removed descendant,
        # not this node's position in its own parent. Re-find the actual owner
        # slot before compacting or another branch can be freed twice.
        ('temp->children[childNumber] = temp->children[temp->numberOfChildren-1];',
         'childNumber = 0;\n'
         '\t\t\t\twhile(childNumber < temp->numberOfChildren && temp->children[childNumber] != currentNode) childNumber++;\n'
         '\t\t\t\tif(childNumber == temp->numberOfChildren) return -999;\n'
         '\t\t\t\ttemp->children[childNumber] = temp->children[temp->numberOfChildren-1];'),
        ('bool fakeSuit[numberOfPlayers][MAX_PLAYERS]', 'bool fakeSuit[numberOfSuits][MAX_PLAYERS]'),
        ('int trickPos = 0;', 'int trickPos = -1;'),
        # Find trick position by players, not suits (5+ player compatibility).
        ('int curpos = 0;\n\tfor(int i = 0; i < numberOfSuits; i++){',
         'trickPos = -1; for(int k=0;k<numberOfPlayers;k++) if(currentTrick[k][0]!=-1) trickPos=k;\n'
         '\tint curpos = 0;\n\tfor(int i = 0; i < numberOfSuits; i++){'),
        # MCTS copies numberOfRounds entries; root buffer must have that capacity.
        ('mctsNode * selectedNode;', 'mctsNode * selectedNode = nullptr;'),
        # A zero UCB value is still a valid candidate. Starting at zero and
        # retaining the last selection can revisit an ancestor or freed node.
        ('currentEval = 0.0;\n\t\t\tfor(int i = 0; i < currentNode->numberOfChildren; i++){',
         'currentEval = -std::numeric_limits<double>::infinity();\n'
         '\t\t\tselectedNode = nullptr;\n'
         '\t\t\tfor(int i = 0; i < currentNode->numberOfChildren; i++){'),
        ('currentNode = selectedNode;',
         'if(selectedNode == nullptr) return -999;\n\t\t\tcurrentNode = selectedNode;'),
        ('mctsNode * selectedChild;', 'mctsNode * selectedChild = nullptr;'),
        ('int sim = 0;', 'int sim = -1;'),
        ('if(currentNode->parent->parent == NULL)',
         'if(currentNode->parent == NULL || currentNode->parent->parent == NULL)'),
        ('for(int i = 0; i < rootPointer->numberOfChildren; i++){\n\t\tif(playableCards[i][0] == childSuit',
         'for(int i = 0; i < numberOfMoves; i++){\n\t\tif(playableCards[i][0] == childSuit'),
        # Completed-round scoring is shared with the real engine.
        ('score[i] += (WIN_CONST + tricksMade[i]*WIN_FACT);',
         'score[i] += (tricksMade[i] == 0 ? WIN_CONST : tricksMade[i]*WIN_FACT);'),
        ('float evalScoreBoard(int player, int curRound, int scoreboard[])',
         'long double evalScoreBoard(int player, int curRound, int scoreboard[])'),
        ('float adjustedScore[numberOfPlayers];', 'long double adjustedScore[numberOfPlayers];'),
        ('float SBvalue = 0.0;', 'long double SBvalue = 0.0;'),
        ('float worstScoreBoardEval =', 'long double worstScoreBoardEval ='),
        ('float bestScoreBoardEval =', 'long double bestScoreBoardEval ='),
        ('float WSEval[numberOfPlayers]', 'long double WSEval[numberOfPlayers]'),
        ('float BSEval[numberOfPlayers]', 'long double BSEval[numberOfPlayers]'),
        ('copyScoreBoard[i] += WIN_CONST + curRound;',
         'copyScoreBoard[i] += max(WIN_CONST, curRound*WIN_FACT);'),
        ('copyScoreBoard[i] += (WIN_CONST + curRound);',
         'copyScoreBoard[i] += max(WIN_CONST, curRound*WIN_FACT);'),
        ('copyScoreBoard[i] -= curRound;', 'copyScoreBoard[i] -= curRound*LOSE_FACT;'),
    ]
    for before, after in replacements:
        if before not in source:
            raise ValueError(f'Pinned upstream patch did not match: {before[:80]}')
        source = source.replace(before, after)
    # Bounded wall-clock search; at least one sample per root action in MC.
    source = source.replace('while(spend < mcBudget)', 'while(spend < mcBudget && !league_expired())')
    source = source.replace('(double)totalTricksMade / (double)mcBudget',
                            '(double)totalTricksMade / (double)max(1, spend)')
    source = source.replace('for(int j = 0; j < budget/numberOfMoves; j++)',
                            'for(int j = 0; j < budget/numberOfMoves && '
                            '((league_counts[i] == 0 && j < 64) || !league_slice_expired(i, numberOfMoves)); j++)')
    # Different action sample counts under deadlines need means, not totals.
    source = source.replace('float simResults[curRound] = {0.0};',
                            'float simResults[curRound] = {0.0}; int league_counts[curRound] = {0};')
    source = source.replace('//set up the copied game state', 'league_counts[i]++;\n\t\t\t//set up the copied game state')
    call = 'playoutRoundRandom(player, curRound, copyGameCards, copyTrickOrder, copyScoreBoard, copyPlayed, copyTrickScore, bids, copySuitBools);'
    if source.count(call) != 1:
        raise ValueError('Pinned MC playout patch no longer matches')
    source = source.replace(call, 'if(!' + call[:-1] + ') { league_counts[i]--; continue; }')
    source = source.replace('\n\t}\n\tint move = 0;\n\tfor (int i = 1; i < curRound; i++){',
                            '\n\t\tif(league_counts[i] == 0) return -1;\n\t}\n\tint move = 0;\n\tfor (int i = 1; i < curRound; i++){')
    source = source.replace('if(simResults[move] < simResults[i])',
                            'if(simResults[move]/max(1,league_counts[move]) < simResults[i]/max(1,league_counts[i]))')
    source = source.replace('int tr = 0;\n\t\tfor(int i = 0; i < numberOfPlayers; i++){',
                            'if(spend >= mcBudget) break;\n\t\tint tr = 0;\n\t\tfor(int i = 0; i < numberOfPlayers; i++){')
    source = source.replace('int childSuit;\n\tint childOrder;',
                            'if(selectedChild == nullptr) return -1;\n\tint childSuit;\n\tint childOrder;')
    # Never pause a headless run waiting for user input on an upstream failure.
    source = source.replace('string t;\n\t\t\tcin >> t;', 'return -999;')
    source = source.replace('string r;\n\t\t\t\t\tcin >> r;', 'return -999;')
    prefix = '''#include <chrono>
static std::chrono::steady_clock::time_point league_start;
static double league_seconds = 0;
static bool league_expired() { return league_seconds > 0 &&
 std::chrono::duration<double>(std::chrono::steady_clock::now()-league_start).count() >= league_seconds; }
static bool league_slice_expired(int i, int n) { return league_seconds > 0 &&
 std::chrono::duration<double>(std::chrono::steady_clock::now()-league_start).count() >= league_seconds*(i+1)/n; }
'''
    return prefix + source


def build_diminishing():
    source = (ASSETS / 'diminishing' / 'main.cc').read_text()
    wrapper = (ROOT / 'ohhell' / 'external_diminishing.cpp').read_text()
    code = patched_diminishing(source) + '\n' + wrapper
    flags = ['-std=gnu++17', '-O2', '-shared', '-static', '-static-libgcc', '-static-libstdc++']
    tag = hashlib.sha256((code + '\n' + repr(flags)).encode()).hexdigest()[:16]
    folder = ASSETS / 'build'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'diminishing_{tag}.dll'
    if not target.exists():
        cpp = folder / f'diminishing_{tag}.cpp'
        cpp.write_text(code)
        compiler = shutil.which('g++')
        if compiler is None:
            raise RuntimeError('g++ is required to build the DiminishingBridge players')
        subprocess.run([compiler, *flags, '-o', str(target), str(cpp)], check=True)
    return target


def verify_assets():
    path = ASSETS / 'manifest.json'
    if not path.exists():
        raise RuntimeError('External bot assets missing; run python -m ohhell.external_league --prepare')
    manifest = json.loads(path.read_text())
    if manifest['revisions'] != {k: list(v) for k, v in REVISIONS.items()}:
        raise ValueError('External revision mismatch')
    for relative, expected in manifest['files'].items():
        if sha256(ASSETS / relative) != expected:
            raise ValueError(f'Changed external asset: {relative}')
    return manifest
