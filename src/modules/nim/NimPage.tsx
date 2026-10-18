import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import katex from "katex";
import "katex/dist/katex.min.css";
import {
  INITIAL_STICKS, chooseAction, createQTable, createVisitTable, greedyActions,
  optimalActions, performTrainingStep, startTrainingEpisode, validActions
} from "./nimLearning";
import type {
  Action, LearningParameters, QTable, TrainingCalculation, TrainingEpisode, VisitTable
} from "./nimLearning";
import "./nim.css";

type Player = "ai" | "human";
type TrainingStatus = "idle" | "running" | "paused";
interface LearningState {
  table: QTable;
  visits: VisitTable;
  games: number;
  wins: number;
}
interface Game {
  remaining: number;
  turn: Player;
  phase: "ready" | "playing" | "finished";
  winner: Player | null;
  history: string[];
}
interface TrainingRun {
  learning: LearningState;
  episode: TrainingEpisode;
  parameters: LearningParameters;
  completed: number;
  target: number;
  calculation: TrainingCalculation | null;
}

const freshLearning = (): LearningState => ({ table: createQTable(), visits: createVisitTable(), games: 0, wins: 0 });
const freshGame = (): Game => ({ remaining: INITIAL_STICKS, turn: "ai", phase: "ready", winner: null, history: [] });
const states = Array.from({ length: INITIAL_STICKS }, (_, i) => INITIAL_STICKS - i);
const actions: Action[] = [1, 2, 3];
const number = (value: number): string => value.toLocaleString("sv-SE", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const math = (formula: string): { __html: string } => ({ __html: katex.renderToString(formula, { throwOnError: false, displayMode: true }) });
const sticksText = (count: number): string => `${count} ${count === 1 ? "pinne" : "pinnar"}`;

function NimPage(): JSX.Element {
  const [learning, setLearning] = useState<LearningState>(freshLearning);
  const [game, setGame] = useState<Game>(freshGame);
  const [firstPlayer, setFirstPlayer] = useState<Player>("ai");
  const [explanation, setExplanation] = useState<"rules" | "strategy" | null>(null);
  const [showChosen, setShowChosen] = useState(true);
  const [showOptimal, setShowOptimal] = useState(false);
  const [showBest, setShowBest] = useState(false);
  const [lastMove, setLastMove] = useState<{ state: number; action: Action } | null>(null);
  const [parameters, setParameters] = useState<LearningParameters>({ alpha: 0.3, gamma: 0.95, epsilon: 0.2 });
  const [gameCount, setGameCount] = useState("20");
  const [realtime, setRealtime] = useState(true);
  const [delay, setDelay] = useState(600);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>("idle");
  const [progress, setProgress] = useState({ completed: 0, target: 0 });
  const [calculation, setCalculation] = useState<TrainingCalculation | null>(null);
  const [trainingMessage, setTrainingMessage] = useState("AI:n börjar utan erfarenhet. Alla Q-värden är 0.");
  const runRef = useRef<TrainingRun | null>(null);
  const training = trainingStatus !== "idle";
  const count = Number(gameCount);
  const validCount = gameCount.trim() !== "" && Number.isInteger(count) && count >= 1 && count <= 100000;

  const makeMove = useCallback((player: Player, action: Action): void => {
    setGame((previous) => {
      if (previous.phase !== "playing" || previous.turn !== player || !validActions(previous.remaining).includes(action)) return previous;
      const remaining = previous.remaining - action;
      return {
        remaining,
        turn: player === "ai" ? "human" : "ai",
        phase: remaining === 0 ? "finished" : "playing",
        winner: remaining === 0 ? (player === "ai" ? "human" : "ai") : null,
        history: [...previous.history, `${player === "ai" ? "AI:n" : "Du"} tog ${sticksText(action)}. ${remaining} kvar.`]
      };
    });
  }, []);

  useEffect(() => {
    if (training || game.phase !== "playing" || game.turn !== "ai") return;
    const timer = window.setTimeout(() => {
      const action = chooseAction(learning.table, game.remaining);
      setLastMove({ state: game.remaining, action });
      makeMove("ai", action);
    }, 750);
    return () => window.clearTimeout(timer);
  }, [game.phase, game.turn, game.remaining, learning.table, makeMove, training]);

  // Each transition includes the opponent's response: s' is the next AI turn.
  const advanceTraining = useCallback((stepLimit: number): boolean => {
    const run = runRef.current;
    if (!run) return true;
    for (let step = 0; step < stepLimit && run.completed < run.target; step += 1) {
      if (run.episode.done) {
        run.episode = startTrainingEpisode(run.learning.games % 2 === 0 ? "ai" : "opponent");
      }
      const result = performTrainingStep(run.learning.table, run.learning.visits, run.episode, run.parameters);
      run.episode = result.episode;
      run.calculation = result.calculation;
      run.learning = { ...run.learning, table: result.table, visits: result.visits };
      if (run.episode.done) {
        run.completed += 1;
        run.learning = {
          ...run.learning,
          games: run.learning.games + 1,
          wins: run.learning.wins + (run.episode.winner === "ai" ? 1 : 0)
        };
      }
    }
    const finished = run.completed === run.target;
    setProgress({ completed: run.completed, target: run.target });
    if (stepLimit === 1 || finished) {
      setLearning(run.learning);
      setCalculation(run.calculation);
      if (run.calculation) setLastMove({ state: run.calculation.state, action: run.calculation.action });
    }
    if (finished) {
      setTrainingStatus("idle");
      setTrainingMessage(`${run.completed.toLocaleString("sv-SE")} nya partier klara. Totalt ${run.learning.games.toLocaleString("sv-SE")} träningspartier. Prova att möta AI:n!`);
      runRef.current = null;
    }
    return finished;
  }, []);

  useEffect(() => {
    if (trainingStatus !== "running") return;
    let timer: number;
    let cancelled = false;
    const tick = (): void => {
      if (cancelled) return;
      const finished = advanceTraining(realtime ? 1 : 500);
      if (!finished) timer = window.setTimeout(tick, realtime ? delay : 0);
    };
    timer = window.setTimeout(tick, realtime ? delay : 0);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [trainingStatus, realtime, delay, advanceTraining]);

  const startTraining = (fromScratch: boolean): void => {
    if (!validCount || training) return;
    const initial = fromScratch ? freshLearning() : learning;
    setLearning(initial);
    setGame(freshGame());
    setCalculation(null);
    setLastMove(null);
    setProgress({ completed: 0, target: count });
    runRef.current = {
      learning: initial,
      episode: startTrainingEpisode(initial.games % 2 === 0 ? "ai" : "opponent"),
      parameters: { ...parameters }, completed: 0, target: count, calculation: null
    };
    setTrainingMessage("Träning pågår mot en slumpmässig motspelare. Spelarna turas om att börja.");
    setTrainingStatus("running");
  };

  const stopTraining = (): void => {
    const run = runRef.current;
    if (run) {
      setLearning(run.learning);
      setCalculation(run.calculation);
      if (run.calculation) setLastMove({ state: run.calculation.state, action: run.calculation.action });
      setTrainingMessage(`Träningen avbröts efter ${run.completed} färdiga partier. Gjorda Q-uppdateringar har sparats, även från det påbörjade partiet.`);
    }
    runRef.current = null;
    setTrainingStatus("idle");
  };

  const startGame = (): void => {
    setLastMove(null);
    setGame({ ...freshGame(), phase: "playing", turn: firstPlayer });
  };
  const toggleExplanation = (panel: "rules" | "strategy"): void => setExplanation((current) => current === panel ? null : panel);
  const activeCalculation = training && realtime ? calculation : null;
  const remaining = activeCalculation ? activeCalculation.nextState : game.remaining;
  const currentState = activeCalculation ? activeCalculation.state : game.remaining;
  const gameStatus = training
    ? trainingStatus === "paused" ? "Träningen är pausad" : "AI:n tränar mot en slumpmässig motspelare"
    : game.phase === "finished"
      ? game.winner === "human" ? "Du vann! AI:n tog sista pinnen." : "AI:n vann! Du tog sista pinnen."
      : game.phase === "ready" ? "Redo för ett parti?" : game.turn === "ai" ? "AI:n väljer ett drag ur Q-tabellen…" : "Din tur. Hur många pinnar tar du?";

  return (
    <div className="container page-flow nim-page">
      <section className="nim-intro">
        <Link className="nim-back" to="/laborationer">← Alla laborationer</Link>
        <div className="nim-eyebrow"><p className="section-eyebrow">Laboration 01 · Förstärkningsinlärning</p><span className="pill">Q-learning</span></div>
        <h1 className="nim-title">Nim – lär AI:n att vinna</h1>
        <p className="nim-lead">Tolv pinnar. Tre möjliga drag. Spela mot AI:n, träna den med Q-learning och upptäck hur erfarenhet blir en strategi.</p>
        <div className="nim-toolbar">
          <button className="btn btn-secondary" aria-expanded={explanation === "rules"} aria-controls="nim-rules" onClick={() => toggleExplanation("rules")}>Spelregler</button>
          <button className="btn btn-secondary" aria-expanded={explanation === "strategy"} aria-controls="nim-strategy" onClick={() => toggleExplanation("strategy")}>Förklara optimal strategi</button>
          <Link className="nim-back" to="/visualiseringar/q-learning">Repetera Q-learning ↗</Link>
        </div>
        {explanation === "rules" && <div className="nim-explanation" id="nim-rules">
          <h2>Den som tar sista pinnen förlorar</h2>
          <p>Spelet börjar med 12 pinnar. Du och AI:n turas om att ta 1, 2 eller 3 pinnar. Du får aldrig ta fler pinnar än det finns kvar. Den som tar den sista pinnen förlorar direkt.</p>
          <p>Välj vem som börjar och starta ett parti. AI:n väljer alltid bland de giltiga drag som har högst Q-värde. Om flera är lika bra väljer den slumpmässigt. Spel mot dig ändrar inte tabellen; inlärningen sker när du tränar AI:n.</p>
        </div>}
        {explanation === "strategy" && <div className="nim-explanation" id="nim-strategy">
          <h2>Så fungerar vinnarstrategin</h2>
          <ol className="nim-strategy-steps">
            <li>
              <h3>En pinne kvar: du förlorar</h3>
              <p>Om det är din tur och bara en pinne finns kvar måste du ta den. Då förlorar du.</p>
            </li>
            <li>
              <h3>Fem pinnar kvar: motspelaren kan tvinga dig att förlora</h3>
              <p>Oavsett hur många pinnar du tar kan motspelaren lämna exakt en pinne till dig:</p>
              <table className="nim-strategy-table">
                <caption className="nim-sr-only">Motspelarens svar när du börjar din tur med fem pinnar kvar.</caption>
                <thead><tr><th scope="col">Du tar</th><th scope="col">Motspelaren tar sedan</th><th scope="col">Kvar när det blir din tur igen</th></tr></thead>
                <tbody>
                  <tr><td>1</td><td>3</td><td>1</td></tr>
                  <tr><td>2</td><td>2</td><td>1</td></tr>
                  <tr><td>3</td><td>1</td><td>1</td></tr>
                </tbody>
              </table>
              <p>Du kan alltså inte undvika förlust om motspelaren svarar rätt.</p>
            </li>
            <li>
              <h3>Nio pinnar kvar: motspelaren kan tvinga dig tillbaka till fem</h3>
              <p>Samma svar fungerar här. Tar du 1 tar motspelaren 3, tar du 2 tar motspelaren 2, och tar du 3 tar motspelaren 1.</p>
              <p>Tillsammans tar ni bort 4 pinnar. När det blir din tur igen finns därför 5 kvar – och vi vet redan varför det är ett förlustläge.</p>
            </li>
            <li>
              <h3>Därför kan den som börjar med tolv pinnar alltid vinna</h3>
              <p>Ta först 3 pinnar och lämna 9 till motspelaren. Svara sedan så att era två drag tillsammans tar 4 pinnar. Då får motspelaren tillbaka turen med först 5 och därefter 1 pinne kvar. Motspelaren måste då ta den sista pinnen och förlorar.</p>
            </li>
            <li>
              <h3>Den generella regeln</h3>
              <p><strong>Försök lämna 1, 5 eller 9 pinnar till motspelaren:</strong> ett antal som är en mer än en multipel av 4. Komplettera sedan motspelarens drag så att era två drag tillsammans tar 4 pinnar.</p>
            </li>
          </ol>
        </div>}
      </section>

      <div className="nim-workspace">
        <div className="nim-left">
          <section className="nim-card" aria-labelledby="nim-game-heading">
            <div className="nim-card-heading"><div><h2 id="nim-game-heading">Spela mot AI:n</h2><p>Ta 1–3 pinnar. Undvik den sista.</p></div><span className="nim-step">01</span></div>
            <div className="nim-board" role="img" aria-label={`${sticksText(remaining)} kvar på spelplanen`}>
              <div className="nim-board-label"><span>Pinnar kvar</span><strong>{remaining}<small> / 12</small></strong></div>
              <div className="nim-sticks" aria-hidden="true">{states.slice().reverse().map((stick) => <span key={stick} className={`nim-stick ${stick > remaining ? "is-gone" : ""}`} />)}</div>
              <p className="nim-board-footer">{training ? "Träningsparti · AI:n + slumpmässig motspelare" : "Sista pinnen = förlust"}</p>
            </div>
            <p className="nim-turn" role="status">{gameStatus}</p>
            <fieldset className="nim-starter" disabled={training || game.phase === "playing"}>
              <legend>Vem börjar?</legend>
              <label><input type="radio" name="nim-starter" checked={firstPlayer === "ai"} onChange={() => setFirstPlayer("ai")} />AI:n</label>
              <label><input type="radio" name="nim-starter" checked={firstPlayer === "human"} onChange={() => setFirstPlayer("human")} />Du</label>
            </fieldset>
            <div className="nim-move-buttons" role="group" aria-label="Välj antal pinnar att ta">{actions.map((action) => <button key={action} className="btn btn-primary" disabled={training || game.phase !== "playing" || game.turn !== "human" || action > game.remaining} onClick={() => makeMove("human", action)}>Ta {action}</button>)}</div>
            <div className="nim-game-actions">
              {game.phase !== "playing" ? <button className="btn btn-secondary" disabled={training} onClick={startGame}>{game.phase === "finished" ? "Spela igen" : "Starta parti"}</button> : <button className="btn btn-secondary" disabled={training} onClick={() => { setGame(freshGame()); setLastMove(null); }}>Avsluta parti</button>}
              <span className="nim-muted">AI:n spelar med sin aktuella Q-tabell.</span>
            </div>
            {game.history.length > 0 && !training && <ol className="nim-history" aria-label="Senaste drag" start={Math.max(1, game.history.length - 3)}>{game.history.slice(-4).map((move, i) => <li key={`${game.history.length}-${i}`}>{move}</li>)}</ol>}
          </section>

          <section className="nim-card" aria-labelledby="nim-training-heading">
            <div className="nim-card-heading"><div><h2 id="nim-training-heading">Träna AI:n</h2><p>Erfarenhet, ett parti i taget.</p></div><span className="nim-step">02</span></div>
            <p className="nim-training-note">AI:n möter en slumpmässig motspelare. De turas om att börja. Vinst ger +1, förlust −1 och övriga drag 0.</p>
            <div className="nim-training-options">
              <label className="nim-field"><span>Antal partier</span><input type="number" min={1} max={100000} step={1} value={gameCount} disabled={training} aria-invalid={!validCount} aria-describedby={!validCount ? "nim-count-error" : undefined} onChange={(event) => setGameCount(event.target.value)} /></label>
              <label className="nim-field"><span>Visningshastighet</span><select value={delay} disabled={training || !realtime} onChange={(event) => setDelay(Number(event.target.value))}><option value={1200}>Lugn</option><option value={600}>Normal</option><option value={150}>Snabb</option></select></label>
            </div>
            {!validCount && <p id="nim-count-error" className="nim-validation" role="alert">Ange ett heltal mellan 1 och 100 000.</p>}
            <div className="nim-parameters">{([
              ["alpha", "α · Inlärning", "Hur mycket ny erfarenhet ändrar Q-värdet."],
              ["gamma", "γ · Framtid", "Hur mycket framtida belöningar räknas."],
              ["epsilon", "ε · Utforskning", "Sannolikheten att prova ett slumpmässigt drag under träning."]
            ] as const).map(([key, title, help]) => <label className="nim-field" key={key}><span>{title}</span><input aria-label={key} title={help} type="number" min={0} max={1} step={0.05} value={parameters[key]} disabled={training} onChange={(event) => { const value = event.target.valueAsNumber; if (Number.isFinite(value)) setParameters((previous) => ({ ...previous, [key]: Math.min(1, Math.max(0, value)) })); }} /><small>{help}</small></label>)}</div>
            <label className="nim-check"><input type="checkbox" checked={realtime} disabled={training} onChange={(event) => setRealtime(event.target.checked)} /><span>Visa Q-tabellens uppdateringar i realtid</span></label>
            <p className="nim-training-note">{realtime ? "Följ Bellman-beräkningen under tabellen. Pausa för att undersöka ett steg." : "Snabbträning: tabellen visas när träningen är klar."}</p>
            <div className="nim-actions">
              {!training ? <><button className="btn btn-primary" disabled={!validCount} onClick={() => startTraining(false)}>{learning.games ? `Träna ${validCount ? count.toLocaleString("sv-SE") : ""} till` : "Starta träning"}</button><button className="btn btn-secondary" disabled={!validCount} onClick={() => startTraining(true)}>Träna från början</button></> : <><button className="btn btn-primary" onClick={() => setTrainingStatus((previous) => previous === "paused" ? "running" : "paused")}>{trainingStatus === "paused" ? "Fortsätt träning" : "Pausa träning"}</button>{trainingStatus === "paused" && realtime && <button className="btn btn-secondary" onClick={() => advanceTraining(1)}>Nästa steg</button>}<button className="btn btn-secondary" onClick={stopTraining}>Avsluta träning</button></>}
            </div>
            {!training && <p className="nim-training-note">”Träna från början” nollställer Q-tabellen och tränar det valda antalet partier. När träning startar avslutas ett eventuellt spelparti.</p>}
            {progress.target > 0 && <div className="nim-training-progress"><progress aria-label="Färdiga träningspartier" value={progress.completed} max={progress.target} /><span>{progress.completed.toLocaleString("sv-SE")} / {progress.target.toLocaleString("sv-SE")} partier</span></div>}
            <p className="nim-training-note" role="status">{trainingMessage}</p>
            <div className="nim-statistics"><div><strong>{learning.games.toLocaleString("sv-SE")}</strong><span>Tränade partier</span></div><div><strong>{learning.wins.toLocaleString("sv-SE")}</strong><span>AI-vinster</span></div><div><strong>{(learning.games - learning.wins).toLocaleString("sv-SE")}</strong><span>AI-förluster</span></div></div>
            <p className="nim-training-note">Träningsresultaten påverkas av utforskning och den slumpmässiga motspelaren. Q-tabellen sparas så länge du stannar i laborationen.</p>
          </section>
        </div>

        <section className="nim-card" aria-labelledby="nim-table-heading">
          <div className="nim-card-heading"><div><h2 id="nim-table-heading">AI:ns Q-tabell</h2><p>Ett tillstånd = antal pinnar vid AI:ns tur.</p></div><span className="nim-step">03</span></div>
          <div className="nim-table-options">
            <label className="nim-check"><input type="checkbox" checked={showChosen} onChange={(event) => setShowChosen(event.target.checked)} /><i className="nim-legend-mark red" aria-hidden="true" /><span>AI:ns senaste drag</span></label>
            <label className="nim-check"><input type="checkbox" checked={showOptimal} onChange={(event) => setShowOptimal(event.target.checked)} /><i className="nim-legend-mark gold" aria-hidden="true" /><span>Optimalt drag (facit)</span></label>
            <label className="nim-check"><input type="checkbox" checked={showBest} onChange={(event) => setShowBest(event.target.checked)} /><i className="nim-legend-mark blue" aria-hidden="true" /><span>Bästa drag hittills</span></label>
          </div>
          <div className="nim-table-scroll"><table className="nim-q-table">
            <caption className="nim-sr-only">Q-värden för 1–12 pinnar. Ogiltiga drag markeras med ett streck. Alla värden är 0 före träning.</caption>
            <thead><tr><th scope="col">Pinnar kvar</th>{actions.map((action) => <th key={action} scope="col">Ta {action}</th>)}</tr></thead>
            <tbody>{states.map((state) => {
              const rowVisited = learning.visits[state].some((visits) => visits > 0);
              const best = rowVisited ? greedyActions(learning.table, state) : [];
              return <tr key={state} className={currentState === state ? "nim-state-active" : ""}><th scope="row">{state}<span className="nim-sr-only"> {state === 1 ? "pinne" : "pinnar"}</span></th>{actions.map((action) => {
                if (action > state) return <td className="nim-invalid" key={action} aria-label="Ogiltigt drag">—</td>;
                const value = learning.table[state][action - 1];
                const marks = [showChosen && lastMove?.state === state && lastMove.action === action ? "red" : null, showOptimal && optimalActions(state).includes(action) ? "gold" : null, showBest && best.includes(action) ? "blue" : null].filter((mark): mark is string => mark !== null);
                const descriptions = marks.map((mark) => mark === "red" ? "AI:ns senaste drag" : mark === "gold" ? "optimalt drag" : "bästa drag hittills");
                return <td key={action} aria-label={`${sticksText(state)}, ta ${action}: Q ${number(value)}${descriptions.length ? `, ${descriptions.join(", ")}` : ""}`} title={`Q(${state}, ${action}) = ${number(value)} · ${learning.visits[state][action - 1]} träningsbesök${descriptions.length ? ` · ${descriptions.join(" · ")}` : ""}`}><span className={`nim-q-value ${value > 0 ? "nim-q-positive" : value < 0 ? "nim-q-negative" : ""}`}><span className="nim-q-number">{number(value)}</span>{marks.map((mark, slot) => <i key={mark} className={`nim-cell-ring ${mark}`} data-slot={slot} aria-hidden="true" />)}</span></td>;
              })}</tr>;
            })}</tbody>
          </table></div>
          <p className="nim-table-note">Högre Q-värde betyder bättre förväntad belöning. Streck = ogiltigt drag. Vid lika högsta värden visas flera blå ramar; obesökta rader får ingen blå ram. Ett oprövat drag har Q = 0 och kan vara radens bästa uppskattning om andra drag är negativa. Guldram saknas i förlustlägena 1, 5 och 9.</p>

          <div className="nim-calculation">
            <h3>Så uppdateras Q-värdet</h3>
            <div className="nim-formula" dangerouslySetInnerHTML={math(String.raw`Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]`)} />
            {calculation ? <>
              <div className="nim-calculation-details"><p><strong>{sticksText(calculation.state)} → AI:n tar {calculation.action}</strong>{calculation.opponentAction !== null ? ` → motspelaren tar ${calculation.opponentAction}.` : "."}</p><p>{calculation.terminal ? "Partiet är slut: framtida Q-värde är 0." : `Nästa AI-tur: ${sticksText(calculation.nextState)}. Bästa Q-värde där: ${number(calculation.maxNextQ)}.`}</p><span className="nim-reward-pill" data-reward={calculation.reward}>Belöning: {calculation.reward > 0 ? "+" : ""}{calculation.reward}</span></div>
              <div className="nim-formula" dangerouslySetInnerHTML={math(String.raw`Q(${calculation.state},${calculation.action}) = ${calculation.oldQ.toFixed(3)} + ${calculation.alpha}\left[${calculation.reward} + ${calculation.gamma}\cdot ${calculation.maxNextQ.toFixed(3)} - (${calculation.oldQ.toFixed(3)})\right] \approx ${calculation.newQ.toFixed(3)}`)} />
              <p className="nim-training-note">{calculation.explored ? "Utforskning: AI:n provade ett slumpmässigt drag (ε)." : "AI:n valde ett drag med högst Q-värde; lika värden avgörs slumpmässigt."} Beräkningen visas avrundad. AI:n använder full precision.</p>
            </> : <p className="nim-training-note">Starta träningen för att se en beräkning med faktiska värden. Med realtidsvisning följer du varje Q-uppdatering här.</p>}
            <p className="nim-training-note">Ett träningssteg omfattar AI:ns drag och motspelarens svar. s′ är nästa tillstånd där AI:n ska välja; γ tillämpas en gång per sådant steg. I sluttillståndet 0 finns inga drag och därför ingen rad i tabellen.</p>
          </div>
        </section>
      </div>

      <section className="nim-card" aria-labelledby="nim-explore-heading">
        <div className="nim-card-heading"><div><p className="section-eyebrow">Undersök själv</p><h2 id="nim-explore-heading">Från slump till strategi</h2></div></div>
        <div className="nim-experiments"><article className="nim-experiment"><h3>1. Spela före träningen</h3><p>Kan du slå en AI som saknar erfarenhet? Visa den röda ramen och följ hur den väljer bland lika Q-värden.</p></article><article className="nim-experiment"><h3>2. Träna och jämför</h3><p>Träna 20 partier. Visa blå och gula ramar. Var stämmer AI:ns val med facit? Träna fler partier och jämför igen.</p></article><article className="nim-experiment"><h3>3. Ändra utforskningen</h3><p>Träna från början med ε = 0 och sedan ε = 1. Hur förändras dragen under träning? Vad händer om α = 0?</p></article></div>
      </section>
    </div>
  );
}

export default NimPage;
