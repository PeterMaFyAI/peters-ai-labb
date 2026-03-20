import { useEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import "./normalisering.css";

const HEIGHTS_CM = [
  183.5, 179.0, 184.5, 190.7, 178.4, 178.4, 191.1, 185.4, 176.7, 183.8,
  176.8, 176.7, 181.7, 166.6, 167.9, 176.1, 172.9, 182.2, 173.6, 170.1
];

const AGES_YEARS = [
  43, 42, 20, 33, 68, 79, 18, 56, 71, 20, 41, 23, 31, 44, 34, 43, 42, 39, 40, 44
];

interface TableRow {
  index: number;
  height: number;
  standardizedHeight: number;
  age: number;
  standardizedAge: number;
}

function standardizeZScore(values: number[]): number[] {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const sigma = Math.sqrt(variance);

  if (sigma === 0) {
    return values.map(() => 0);
  }

  return values.map((value) => (value - mean) / sigma);
}

function buildSharedRange(values: number[]): [number, number] {
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const span = maxValue - minValue;

  if (span === 0) {
    return [minValue - 1, maxValue + 1];
  }

  const padding = Math.max(span * 0.08, 0.2);
  return [minValue - padding, maxValue + padding];
}

function renderMath(expression: string): { __html: string } {
  return {
    __html: katex.renderToString(expression, {
      throwOnError: false,
      displayMode: true
    })
  };
}

function NormaliseringPage(): JSX.Element {
  const leftPlotRef = useRef<HTMLDivElement | null>(null);
  const rightPlotRef = useRef<HTMLDivElement | null>(null);
  const plotlyRef = useRef<any>(null);
  const [showNormalized, setShowNormalized] = useState(false);

  const standardizedHeights = useMemo(() => standardizeZScore(HEIGHTS_CM), []);
  const standardizedAges = useMemo(() => standardizeZScore(AGES_YEARS), []);

  const tableRows = useMemo<TableRow[]>(
    () =>
      HEIGHTS_CM.map((height, index) => ({
        index: index + 1,
        height,
        standardizedHeight: standardizedHeights[index],
        age: AGES_YEARS[index],
        standardizedAge: standardizedAges[index]
      })),
    [standardizedAges, standardizedHeights]
  );

  const displayedHeights = showNormalized
    ? tableRows.map((row) => row.standardizedHeight)
    : tableRows.map((row) => row.height);
  const displayedAges = showNormalized
    ? tableRows.map((row) => row.standardizedAge)
    : tableRows.map((row) => row.age);

  const normalizationFormulaHtml = useMemo(
    () => renderMath(String.raw`z=\frac{x-\mu}{\sigma}`),
    []
  );

  useEffect(() => {
    let cancelled = false;

    const renderPlots = async (): Promise<void> => {
      const leftPlotElement = leftPlotRef.current;
      const rightPlotElement = rightPlotRef.current;

      if (!leftPlotElement || !rightPlotElement) {
        return;
      }

      if (!plotlyRef.current) {
        const plotlyModule = await import("plotly.js-dist-min");
        if (cancelled) {
          return;
        }
        plotlyRef.current = plotlyModule.default;
      }

      const Plotly = plotlyRef.current;
      const combinedDisplayedValues = [...displayedHeights, ...displayedAges];
      const leftXRange = showNormalized ? buildSharedRange(combinedDisplayedValues) : [0, 200];
      const rightAxisRange = showNormalized
        ? buildSharedRange(combinedDisplayedValues)
        : [0, 200];

      const leftTraces = [
        {
          type: "scatter",
          mode: "markers",
          name: "Längd (cm)",
          x: displayedHeights,
          y: tableRows.map(() => 2),
          marker: {
            color: "#1f6feb",
            size: 10
          },
          hovertemplate: "x=%{x:.3f}<extra>Längd</extra>"
        },
        {
          type: "scatter",
          mode: "markers",
          name: "Ålder (år)",
          x: displayedAges,
          y: tableRows.map(() => 6),
          marker: {
            color: "#bf2f3a",
            size: 10
          },
          hovertemplate: "x=%{x:.3f}<extra>Ålder</extra>"
        }
      ];

      const leftLayout = {
        margin: { l: 56, r: 20, t: 16, b: 42 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#f8fbfe",
        xaxis: {
          title: showNormalized ? "Värde (standard score)" : "Värde",
          range: leftXRange,
          zeroline: false
        },
        yaxis: {
          range: [0, 8],
          tickvals: [2, 6],
          ticktext: ["Längd", "Ålder"],
          zeroline: false
        },
        legend: {
          orientation: "h",
          y: 1.12
        }
      };

      const rightTrace = {
        type: "scatter",
        mode: "markers",
        name: "Datapunkter",
        x: displayedHeights,
        y: displayedAges,
        customdata: tableRows.map((row) => row.index),
        marker: {
          color: "#121212",
          size: 10
        },
        hovertemplate: "Index=%{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
      };

      const rightLayout = {
        margin: { l: 56, r: 20, t: 16, b: 48 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#f8fbfe",
        xaxis: {
          title: showNormalized ? "Längd (standard score)" : "Längd (cm)",
          range: rightAxisRange,
          zeroline: false
        },
        yaxis: {
          title: showNormalized ? "Ålder (standard score)" : "Ålder (år)",
          range: rightAxisRange,
          zeroline: false,
          scaleanchor: "x",
          scaleratio: 1
        }
      };

      await Promise.all([
        Plotly.react(leftPlotElement, leftTraces, leftLayout, {
          responsive: true,
          displaylogo: false
        }),
        Plotly.react(rightPlotElement, [rightTrace], rightLayout, {
          responsive: true,
          displaylogo: false
        })
      ]);
    };

    void renderPlots();

    return () => {
      cancelled = true;
    };
  }, [displayedAges, displayedHeights, showNormalized]);

  useEffect(() => {
    return () => {
      if (plotlyRef.current) {
        if (leftPlotRef.current) {
          plotlyRef.current.purge(leftPlotRef.current);
        }
        if (rightPlotRef.current) {
          plotlyRef.current.purge(rightPlotRef.current);
        }
      }
    };
  }, []);

  return (
    <div className="container page-flow">
      <section className="section normalisering-header">
        <div>
          <h1 className="normalisering-page-title">Normalisering av egenskaper</h1>
          <p className="normalisering-page-lead">
            Här jämför vi längd och ålder före och efter normalisering för att se varför skalning
            är viktig när olika egenskaper har helt olika numeriska intervall.
          </p>
        </div>
        <button
          className="btn btn-primary"
          type="button"
          onClick={() => setShowNormalized((previousValue) => !previousValue)}
        >
          {showNormalized ? "Visa vanliga värden i diagrammen" : "Visa normaliserade värden i diagrammen"}
        </button>
      </section>

      <main className="normalisering-layout">
        <section className="section">
          <h2>Värden på varsin nivå</h2>
          <p className="lead">
            Blå punkter (längd) ligger på y = 2 och röda punkter (ålder) på y = 6.
          </p>
          <div ref={leftPlotRef} className="normalisering-plot" />
        </section>

        <section className="section">
          <h2>Längd mot ålder</h2>
          <p className="lead">
            Svarta punkter visar varje persons längd och ålder som ett punktmoln.
          </p>
          <div ref={rightPlotRef} className="normalisering-plot" />
        </section>

        <section className="section">
          <h2>Tabell med normaliserade värden</h2>
          <div className="normalisering-table-wrap">
            <table className="normalisering-table">
              <thead>
                <tr>
                  <th>Nr</th>
                  <th>Längd (cm)</th>
                  <th>
                    Normaliserad
                    <br />
                    längd
                  </th>
                  <th>Ålder (år)</th>
                  <th>
                    Normaliserad
                    <br />
                    ålder
                  </th>
                </tr>
              </thead>
              <tbody>
                {tableRows.map((row) => (
                  <tr key={row.index}>
                    <td>{row.index}</td>
                    <td>{row.height.toFixed(1)}</td>
                    <td>{row.standardizedHeight.toFixed(4)}</td>
                    <td>{row.age.toFixed(0)}</td>
                    <td>{row.standardizedAge.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="section normalisering-explanation">
          <h2>Kort förklaring</h2>
          <p className="lead">
            Här använder vi standard score. Varje värde centreras kring medelvärdet och skalas med
            standardavvikelsen, så att egenskaper med olika enheter blir jämförbara.
          </p>
          <div
            className="linear-regression-math-block normalisering-math-block"
            dangerouslySetInnerHTML={normalizationFormulaHtml}
          />
          <p className="lead">
            Efter standardisering får varje egenskap medelvärde nära 0 och spridning nära 1, vilket
            ofta gör träning och avståndsberäkningar mer balanserade.
          </p>
        </section>
      </main>
    </div>
  );
}

export default NormaliseringPage;
