// Native, editable one-slide research figure from measured validation values.
// Runtime paths are supplied by Codex load_workspace_dependencies; no installs.
import fs from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [inputPath, buildDir, finalPath] = process.argv.slice(2);
if (!inputPath || !buildDir || !finalPath) throw new Error("Expected state JSON, private build directory, final PPTX path");
const skillDir = process.env.PRESENTATIONS_SKILL_DIR;
const workspaceDir = path.dirname(path.resolve(buildDir));
const helpers = await import(pathToFileURL(path.join(skillDir, "container_tools/runtime_helpers.mjs")).href);
const { Presentation, PresentationFile } = await helpers.importRuntimeModule("@oai/artifact-tool");
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
const font = resolvePresentationFont({ fontFamily: "Arial" });
const fontPolicy = { basis: "design", families: [font] };
const state = JSON.parse(await fs.readFile(inputPath, "utf8"));
const methods = [
  ["frozen", "Frozen / Fixed", "#1f77b4", "square"],
  ["ddpo_only", "DDPO-only", "#ff7f0e", "circle"],
  ["rule_rsi", "Rule-based RSI", "#2ca02c", "diamond"],
  ["rl_rsi", "RL-RSI", "#d62728", "triangle"],
];
const rawSeries = methods.map(([key, name, color, symbol]) => ({
  name, values: [0, 1, 2, 3].map(round => state.validation?.[key]?.find(point => point.round === round)?.common_reward ?? null),
  line: { style: "solid", fill: color, width: key === "rule_rsi" ? 2 : 3 },
  marker: { symbol, size: key === "rule_rsi" ? 6 : 8, fill: color },
}));
// Excel stores at most 15 significant decimal digits. Round only the figure's
// editable workbook values to nine decimal places; preserve the raw values in
// the sidecar and authoritative experiment JSON/CSV.
const series = rawSeries.map(item => ({ ...item,
  values: item.values.map(value => value === null ? null : Number(value.toFixed(9))),
}));
if (state.status === "COMPLETED" && series.some(item => item.values.some(value => value === null))) {
  throw new Error("A completed figure requires all four measured validation points for every arm");
}
const values = series.flatMap(item => item.values.filter(value => value !== null));
const low = Math.floor(Math.min(...values) / .025) * .025 - .025;
const high = Math.max(0, Math.ceil(Math.max(...values) / .025) * .025 + .025);
await fs.mkdir(buildDir, { recursive: true });
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = presentation.slides.add();
slide.background.fill = "#ffffff";
const chart = slide.charts.add("line", {
  position: { left: 45, top: 25, width: 1190, height: 602 },
  title: "Validation common reward", titleTextStyle: { fontSize: 34, fill: "#111827", bold: true },
  categories: ["0", "1", "2", "3"], series,
  hasLegend: true, legend: { position: "bottom", overlay: false, textStyle: { fontSize: 20, fill: "#111827" } },
  lineOptions: { grouping: "standard", smooth: false },
  xAxis: { title: { text: "Completed round", textStyle: { fontSize: 23, fill: "#111827" } },
    textStyle: { fontSize: 20, fill: "#374151" }, position: "bottom", tickLabelPosition: "low",
    line: { style: "solid", fill: "#6b7280", width: 1 } },
  yAxis: { title: { text: "2 × CLIP cosine − LPIPS", textStyle: { fontSize: 23, fill: "#111827" } },
    numberFormatCode: "0.000", min: low, max: high, majorUnit: .025,
    textStyle: { fontSize: 20, fill: "#374151" }, position: "left",
    majorGridlines: { style: "solid", fill: "#e5e7eb", width: 1 },
    line: { style: "solid", fill: "#6b7280", width: 1 } },
  chartFill: "#ffffff", plotAreaFill: "#ffffff", displayBlanksAs: "gap",
});
applyPresentationChartFont(chart, { fontFamily: font });
const caption = slide.shapes.add({ geometry: "textbox", position: { left: 75, top: 638, width: 1130, height: 64 },
  fill: "none", line: { fill: "none", width: 0 } });
caption.text = "One training seed · development validation (16 images) · common fixed evaluator\nFrozen and Fixed share observations; coincident measured curves are preserved";
caption.text.style = { fontSize: 19, typeface: font, color: "#4b5563", alignment: "center" };
slide.speakerNotes.text = `Source: ${path.resolve(inputPath)}. Pilot run pilot_run_20261104T030649Z_s42. ` +
  "Measured values are embedded as a native chart and workbook snapshot. No ASR, Delta Clean or semantic-correctness claim. " +
  "Fixed is the user-approved Frozen equivalence control. The common evaluator is 2*CLIP cosine - calibrated AlexNet LPIPS for every arm. " +
  "Data from historically exposed VQAv2 development images. Source-image uncertainty is not training-seed replication. " +
  "LPIPS contract: https://github.com/richzhang/PerceptualSimilarity";
const png = await presentation.export({ slide, format: "png", scale: 1 });
await fs.writeFile(path.join(buildDir, "figure_preview.png"), new Uint8Array(await png.arrayBuffer()));
const candidate = path.join(buildDir, "figure_candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidate);
await fs.writeFile(path.join(buildDir, "chart_values.json"), JSON.stringify({ status: state.status,
  raw_full_precision_series: rawSeries, plotted_series: series, plot_decimal_places: 9,
  maximum_plot_rounding_error: Math.max(...rawSeries.flatMap((item, index) => item.values.map((value, j) =>
    value === null ? 0 : Math.abs(value - series[index].values[j])))),
}, null, 2));
if (state.status !== "COMPLETED") {
  console.log(JSON.stringify({ preview_only: true, status: state.status, draft: candidate }));
} else {
  await fs.mkdir(path.dirname(finalPath), { recursive: true });
  const receiptPath = path.join(buildDir, "figure_validation.json");
  await fs.mkdir(path.join(buildDir, "validated_output"), { recursive: true });
  const validatedPath = path.join(buildDir, "validated_output", "validated_final.pptx");
  await finalizePresentation({ workspaceDir, candidatePath: candidate, finalPath: validatedPath,
    pythonExecutable: process.env.RUNTIME_PYTHON,
    integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit"],
    explicitTotalSlideCount: 1, requiredNativeChartOwnerSlides: [1], requiredNativeTableOwnerSlides: [],
    materializeLiteralChartWorkbooks: true, fontPolicy, verifyArtifactToolImport: true, receiptPath,
  });
  await fs.copyFile(validatedPath, finalPath, constants.COPYFILE_EXCL);
  console.log(JSON.stringify({ final: finalPath, validation: receiptPath, preview: path.join(buildDir, "figure_preview.png") }));
}
