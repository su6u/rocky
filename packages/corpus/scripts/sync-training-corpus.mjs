import { writeFrozenTrainerExport } from "../dist/write-frozen-export.js"

const frozen = writeFrozenTrainerExport()
console.log(
  `export: ${frozen.manifest.rowCount} train rows -> ${frozen.trainExportPath}; holdout -> ${frozen.holdoutExportPath}`,
)
