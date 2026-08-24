/**
 * Validator respons API beserta pengunci tipenya.
 *
 * Tipe API tidak pernah ditulis tangan di sini (R3.7). Bentuk yang dipercaya
 * berasal dari `src/types/api.d.ts`, yang dibangkitkan dari `/openapi.json`
 * milik FastAPI - dan dokumen itu sendiri diturunkan dari skema Pydantic milik
 * modul AI, yaitu sumber kebenaran tunggal menurut PROJECT.md 9.1.
 *
 * Skema Zod di bawah bukan salinan kedua dari kontrak itu. Ia penjaga saat
 * jalan: tipe TypeScript hilang sesudah kompilasi, sehingga respons yang
 * bentuknya berubah diam-diam akan lolos sampai menjadi `undefined` di tengah
 * penyajian. Zod membuat ketidakcocokan itu muncul sebagai satu galat yang
 * jelas di tempat respons diterima.
 *
 * Supaya penjaga itu sendiri tidak menyimpang dari kontrak, setiap skema diikat
 * ke tipe bangkitannya lewat `AssertMutuallyAssignable` di bagian bawah berkas.
 * Menambah, menghapus, atau mengganti tipe satu medan pada modul AI akan
 * memunculkan galat kompilasi di sini, bukan bug diam-diam saat demo.
 */

import { z } from "zod";

import type { components } from "@/types/api";

type Schemas = components["schemas"];

export const bboxSchema = z.object({
  x: z.number(),
  y: z.number(),
  w: z.number(),
  h: z.number(),
});

export const defectSchema = z.object({
  type: z.string(),
  label: z.string(),
  bbox: bboxSchema,
  confidence: z.number(),
  area_pct: z.number().nullable().optional(),
});

export const anomalySchema = z.object({
  score: z.number(),
  threshold: z.number(),
  exceeded: z.boolean(),
  heatmap_base64: z.string().nullable().optional(),
});

export const decisionSchema = z.object({
  calibrated_probability: z.number(),
  prediction_set: z.array(z.string()),
  severity: z.number(),
  conformal_alpha: z.number(),
});

export const inspectionResultSchema = z.object({
  verdict: z.enum(["PASS", "REJECT", "REVIEW"]),
  reason: z.string(),
  confidence: z.number().nullable().optional(),
  batch_code: z.string().nullable().optional(),
  defects: z.array(defectSchema).optional(),
  defect_area_pct: z.number(),
  anomaly: anomalySchema.nullable().optional(),
  decision: decisionSchema.nullable().optional(),
  annotated_image_base64: z.string(),
  model_version: z.string(),
  latency_ms: z.number(),
});

export const healthStatusSchema = z.object({
  status: z.enum(["ok", "degraded"]),
  components: z.record(z.string(), z.boolean()),
  detail: z.string().nullable().optional(),
});

export const modelInfoSchema = z.object({
  model_name: z.string(),
  version: z.string(),
  dataset: z.string(),
  trained_at: z.string().nullable().optional(),
  metrics: z.record(z.string(), z.number()),
  components: z.record(z.string(), z.boolean()),
});

export const sampleImageSchema = z.object({
  name: z.string(),
  url: z.string(),
});

export const sampleListSchema = z.array(sampleImageSchema);

/**
 * Pengunci kontrak.
 *
 * `Mutual` hanya meloloskan sepasang tipe yang saling terisi: A harus dapat
 * dipakai di tempat B diminta, dan sebaliknya. Satu arah saja tidak cukup -
 * skema yang kehilangan satu medan tetap lolos pemeriksaan satu arah.
 *
 * Tipe yang diekspor di bawah karena itu bukan sekadar alias. Masing-masing
 * melewati penguncian ini lebih dulu, sehingga menambah, menghapus, atau
 * mengganti tipe satu medan pada skema Pydantic modul AI akan memunculkan galat
 * kompilasi di berkas ini - bukan medan yang diam-diam menjadi `undefined` di
 * tengah penyajian.
 */
// Parameter ketiga bukan hiasan: menuliskan syaratnya sebagai `A extends B,
// B extends A` membuat TypeScript menolaknya sebagai batasan melingkar.
type Mutual<A extends B, B extends C, C = A> = A;

export type BBox = Mutual<Schemas["BBox"], z.infer<typeof bboxSchema>>;
export type Defect = Mutual<Schemas["Defect"], z.infer<typeof defectSchema>>;
export type AnomalyResult = Mutual<Schemas["AnomalyResult"], z.infer<typeof anomalySchema>>;
export type DecisionDetail = Mutual<
  Schemas["DecisionDetail"],
  z.infer<typeof decisionSchema>
>;
export type InspectionResult = Mutual<
  Schemas["InspectionResult"],
  z.infer<typeof inspectionResultSchema>
>;
export type HealthStatus = Mutual<Schemas["HealthStatus"], z.infer<typeof healthStatusSchema>>;
export type ModelInfo = Mutual<Schemas["ModelInfo"], z.infer<typeof modelInfoSchema>>;
export type SampleImage = Mutual<Schemas["SampleImage"], z.infer<typeof sampleImageSchema>>;
export type VerdictLabel = InspectionResult["verdict"];
