import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ANSWER_LENGTH, PROMPTS, type PromptId } from "./languageModelData";
import {
  TEMPERATURES, advanceGeneration, candidatesAt, joinTokens, percentageLabels,
  startGeneration, type TemperatureId
} from "./languageModel";
import "./languageModel.css";

function Arrow({ active = false }: { active?: boolean }): JSX.Element {
  return <div className={`llm-arrow${active ? " is-active" : ""}`} aria-hidden="true">
    <svg viewBox="0 0 40 24"><path d="M2 12h32m-8-8 8 8-8 8" /></svg>
  </div>;
}

function LanguageModelPage(): JSX.Element {
  const [promptId, setPromptId] = useState<PromptId>("capital");
  const [temperatureId, setTemperatureId] = useState<TemperatureId>("narrator");
  const [showProbabilities, setShowProbabilities] = useState(false);
  const [generation, setGeneration] = useState(startGeneration);
  const advanceLock = useRef(false);
  const prompt = PROMPTS.find(({ id }) => id === promptId)!;
  const temperature = TEMPERATURES.find(({ id }) => id === temperatureId)!;
  const complete = generation.tokens.length === ANSWER_LENGTH;
  const prepared = generation.phase === "prepared";
  const sampled = generation.phase === "sampled";
  const started = generation.phase !== "ready";
  const latestDraw = sampled ? generation.draws[generation.draws.length - 1] : undefined;
  const drawTemperature = latestDraw && TEMPERATURES.find(({ id }) => id === latestDraw.temperature);
  const candidates = useMemo(
    () => candidatesAt(promptId, generation.input, temperatureId).sort((a, b) => b.probability - a.probability),
    [promptId, generation.input, temperatureId]
  );
  const percentages = percentageLabels(candidates);
  const tokenNumber = prepared ? generation.tokens.length + 1 : Math.max(1, generation.tokens.length);

  let status = "Börja med att mata in prompten i språkmodellen.";
  if (prepared) status = `Hela texten är inmatad. Nu finns fem möjliga alternativ för token ${tokenNumber}. Slumpa fram ett av dem.`;
  if (sampled) status = `Tokenet ”${latestDraw?.token}” valdes slumpmässigt. Nästa steg matar in prompten tillsammans med alla genererade tokens.`;
  if (complete) status = "Fem tokens är genererade! Börja om för att dra ett nytt svar, eller prova en annan temperatur.";

  function advance(): void {
    // A double-click must not skip straight past the separate input/output stages.
    if (advanceLock.current || complete) return;
    advanceLock.current = true;
    setGeneration(advanceGeneration(generation, promptId, temperatureId));
    window.setTimeout(() => { advanceLock.current = false; }, 250);
  }

  function reset(): void {
    advanceLock.current = false;
    setGeneration(startGeneration());
  }

  return <div className="container page-flow llm-page">
    <header className="llm-intro">
      <Link className="llm-back" to="/laborationer">← Alla laborationer</Link>
      <p className="section-eyebrow">Laboration 02 · Språkmodeller</p>
      <h1>En token i taget</h1>
      <p>Hur blir en prompt till ett svar? Följ texten genom språkmodellen och upptäck hur slump och temperatur påverkar nästa token.</p>
    </header>

    <section className="section llm-lab" aria-label="Språkmodellens laboration">
      <div className="llm-settings">
        <label className="llm-prompt-picker">
          <span className="llm-label">Välj en prompt</span>
          <select value={promptId} onChange={(event) => { setPromptId(event.target.value as PromptId); reset(); }}>
            {PROMPTS.map(({ id, label, text }) => <option value={id} key={id}>{label}: {text}</option>)}
          </select>
        </label>
        <label className="llm-probability-toggle">
          <input type="checkbox" checked={showProbabilities} onChange={(event) => setShowProbabilities(event.target.checked)} />
          <span>Visa sannolikheter</span>
        </label>
      </div>

      <fieldset className="llm-temperature">
        <legend>Temperatur <span>Hur mycket får modellen överraska?</span></legend>
        <div className="llm-temperature-options">
          {TEMPERATURES.map((item) => <button
            key={item.id} type="button" aria-pressed={temperatureId === item.id}
            className={`llm-temperature-button llm-temperature-${item.id}`}
            onClick={() => setTemperatureId(item.id)}
          >
            <span className="llm-temperature-symbol" aria-hidden="true">{item.symbol}</span>
            <span><strong>{item.name}</strong><small>{item.description}</small></span>
            <span className="llm-temperature-value">T = {item.value.toLocaleString("sv-SE")}</span>
          </button>)}
        </div>
      </fieldset>

      <div className={`llm-flow${showProbabilities ? " with-probabilities" : ""}`}>
        <div className={`llm-input-card${prepared ? " is-active" : ""}`}>
          <div className="llm-panel-title"><span className="llm-stage-number">1</span><h2>{started ? "Inmatad text" : "Text att mata in"}</h2></div>
          <span className="llm-text-label">Prompt</span>
          <p className="llm-prompt-text">{prompt.text}</p>
          <div className="llm-context-answer">
            <span className="llm-text-label">Tidigare genererade tokens</span>
            {generation.input.length > 0
              ? <p className="llm-input-tokens">{generation.input.map((token, index) => <span key={index}>{token}</span>)}</p>
              : <p className="llm-no-tokens">Inga ännu — bara prompten.</p>}
          </div>
          <div className="llm-context-footnote">Hela texten skickas in tillsammans.</div>
        </div>

        <Arrow active={prepared} />

        <div className={`llm-model${prepared ? " is-processing" : ""}`}>
          <div className="llm-model-ports" aria-hidden="true"><i /><i /><i /></div>
          <span className="llm-model-name">LLM</span>
          <span className="llm-model-subtitle">Språkmodell</span>
          <span className="llm-model-state"><i />{prepared ? "Fördelning klar" : sampled ? "Token vald" : "Redo"}</span>
        </div>

        <Arrow active={sampled} />

        <div className="llm-output-card">
          <div className="llm-panel-title"><span className="llm-stage-number">2</span><h2>Nästa token</h2><span className="llm-token-index">{tokenNumber} / {ANSWER_LENGTH}</span></div>
          <div className={`llm-output-token${sampled ? " has-token" : ""}`} key={`${generation.tokens.length}-${generation.phase}`}>
            {latestDraw ? latestDraw.token : <span aria-label="Token har inte valts ännu">?</span>}
          </div>
          <p className="llm-output-caption">{sampled ? "Ett slumpval ur fördelningen" : prepared ? "Fem möjligheter. Ett slumpval." : "Ett nytt token kommer ut här."}</p>

          {showProbabilities && <div className="llm-probabilities">
            <div className="llm-table-heading"><h3>Möjliga nästa tokens</h3><span>T = {temperature.value.toLocaleString("sv-SE")}</span></div>
            {started ? <table>
              <caption className="llm-sr-only">Sannolikheter för nästa token vid aktuell temperatur. Det valda tokenet är markerat.</caption>
              <thead><tr><th scope="col">Token</th><th scope="col">Sannolikhet</th></tr></thead>
              <tbody>{candidates.map((candidate, index) => {
                const selected = latestDraw?.token === candidate.token;
                return <tr key={candidate.token} className={selected ? "is-selected" : ""}>
                  <th scope="row"><span className={`llm-candidate-word${selected ? " is-selected" : ""}`}>{candidate.token}</span>{selected && <span className="llm-sr-only"> (vald)</span>}</th>
                  <td><span className="llm-probability-bar" aria-hidden="true"><i style={{ width: `${candidate.probability * 100}%` }} /></span><span className="llm-probability-value">{percentages[index]}</span></td>
                </tr>;
              })}</tbody>
            </table> : <p className="llm-probability-placeholder">Mata in texten för att se de fem alternativen.</p>}
            {latestDraw && drawTemperature && drawTemperature.id !== temperatureId && <p className="llm-temperature-note">Valet gjordes vid T = {drawTemperature.value.toLocaleString("sv-SE")}. Tabellen visar nu T = {temperature.value.toLocaleString("sv-SE")}; tokenet lottas inte om.</p>}
          </div>}
        </div>
      </div>

      <div className="llm-step-controls">
        <div className="llm-step-buttons">
          <button className="btn btn-primary llm-next" onClick={advance} disabled={complete}>
            <span aria-hidden="true">{prepared ? "⚄" : "→"}</span>
            {complete ? "Svaret är färdigt" : prepared ? "Slumpa nästa token" : generation.tokens.length > 0 ? "Mata in hela texten igen" : "Mata in prompten"}
          </button>
          <button className="btn btn-ghost" onClick={reset}>↻ Börja om</button>
        </div>
        <p className="llm-status" role="status" aria-live="polite" aria-atomic="true">{status}</p>
      </div>

      <div className={`llm-answer${complete ? " is-complete" : ""}`}>
        <div className="llm-answer-heading"><h2>Modellens svar</h2><span>{generation.tokens.length} av {ANSWER_LENGTH} tokens</span></div>
        <ol className="llm-answer-tokens" aria-label="Genererade tokens">
          {Array.from({ length: ANSWER_LENGTH }, (_, index) => {
            const draw = generation.draws[index];
            return <li key={index} className={draw ? "is-filled" : ""} aria-label={`Token ${index + 1}: ${draw?.token ?? "inte genererat"}`}>
              <span>{draw?.token ?? "·"}</span><small>{index + 1}</small>
            </li>;
          })}
        </ol>
        <p className="llm-answer-text">{generation.tokens.length > 0 ? joinTokens(generation.tokens) : "Ditt svar växer fram här."}</p>
        <p className="llm-answer-note">Ett ord, ett tal eller ett skiljetecken räknas som ett token här. Vi stannar efter fem tokens, även om meningen inte är färdig.</p>
      </div>
    </section>

    <section className="llm-investigate" aria-labelledby="llm-investigate-heading">
      <div className="llm-investigate-header"><p className="section-eyebrow">Prova själv</p><h2 id="llm-investigate-heading">Samma prompt. Samma svar?</h2></div>
      <div className="llm-experiments">
        <article><span>01</span><h3>Leta efter slumpen</h3><p>Börja om med samma prompt och temperatur några gånger. Väljs alltid det mest sannolika tokenet?</p></article>
        <article><span>02</span><h3>Släpp fram konstnären</h3><p>Visa sannolikheterna och växla mellan Revisorn och Konstnären. Vad händer med fördelningen?</p></article>
        <article><span>03</span><h3>Följ hela texten</h3><p>Jämför den inmatade texten före och efter ett nytt token. Varför behöver modellen se sitt eget svar?</p></article>
      </div>
    </section>

    <details className="llm-about">
      <summary>Om simuleringen och temperatur</summary>
      <div>
        <p>Det här är en simulering. Alla sannolikheter är uppskattade för undervisning och hämtas ur en förberedd databas som tar hänsyn till prompten och samtliga genererade tokens. Den omfattar 2 343 inmatningskontexter och 9 375 möjliga svar med fem tokens. Ingen språkmodell behöver laddas ner.</p>
        <p>I en verklig språkmodell beräknar ett neuralt nätverk sannolikheter över en mycket större uppsättning tokens. Ett token kan då också vara en del av ett ord. Här är uppsättningen förenklad till fem alternativ per steg.</p>
        <p>Temperaturen ändrar fördelningen enligt <span className="llm-formula">p<sub>T</sub>(i) = p(i)<sup>1/T</sup> / Σ<sub>j</sub> p(j)<sup>1/T</sup></span>. Låg temperatur förstärker skillnaderna; hög temperatur jämnar ut dem. Även Revisorn kan dra ett osannolikt token. Högre temperatur ger mer variation och kan också ge konstiga eller felaktiga svar.</p>
      </div>
    </details>
  </div>;
}

export default LanguageModelPage;
