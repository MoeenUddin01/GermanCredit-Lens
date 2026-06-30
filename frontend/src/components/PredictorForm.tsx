"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowRight,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Info,
  Shield,
  TrendingUp,
  Activity,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Card } from "@/components/ui/Card";
import { usePredictor } from "@/hooks/usePredictor";

const CATEGORICAL_OPTIONS: Record<string, string[]> = {
  status_checking: ["A11 (< 0 DM)", "A12 (0–200 DM)", "A13 (≥ 200 DM)", "A14 (no checking)"],
  credit_history: [
    "A30 (no credits taken)",
    "A31 (all paid back duly)",
    "A32 (existing credits paid back)",
    "A33 (delay in past)",
    "A34 (critical account)",
  ],
  purpose: [
    "A40 (car new)",
    "A41 (car used)",
    "A42 (furniture)",
    "A43 (radio/tv)",
    "A44 (domestic appliances)",
    "A45 (repairs)",
    "A46 (education)",
    "A47 (vacation)",
    "A48 (retraining)",
    "A49 (business)",
    "A410 (others)",
  ],
  savings_bonds: [
    "A61 (< 100 DM)",
    "A62 (100–500 DM)",
    "A63 (500–1000 DM)",
    "A64 (≥ 1000 DM)",
    "A65 (unknown/none)",
  ],
  employment_since: [
    "A71 (unemployed)",
    "A72 (< 1 year)",
    "A73 (1–4 years)",
    "A74 (4–7 years)",
    "A75 (≥ 7 years)",
  ],
  personal_status_sex: [
    "A91 (male: divorced/separated)",
    "A92 (female: divorced/separated/married)",
    "A93 (male: single)",
    "A94 (male: married/widowed)",
    "A95 (female: single)",
  ],
  other_debtors: ["A101 (none)", "A102 (co-applicant)", "A103 (guarantor)"],
  property: [
    "A121 (real estate)",
    "A122 (building society / life insurance)",
    "A123 (car or other)",
    "A124 (unknown/none)",
  ],
  other_installment: ["A141 (bank)", "A142 (stores)", "A143 (none)"],
  housing: ["A151 (rent)", "A152 (own)", "A153 (for free)"],
  job: [
    "A171 (unemployed/unskilled)",
    "A172 (unskilled resident)",
    "A173 (skilled/self-employed)",
    "A174 (executive/self-employed highly qualified)",
  ],
  telephone: ["A191 (none)", "A192 (yes, registered)"],
  foreign_worker: ["A201 (yes)", "A202 (no)"],
};

const ALL_FIELDS = [
  { key: "status_checking", label: "Checking Account", type: "select" as const, category: "Applicant Profile" },
  { key: "credit_history", label: "Credit History", type: "select" as const, category: "Credit History" },
  { key: "purpose", label: "Loan Purpose", type: "select" as const, category: "Loan Details" },
  { key: "credit_amount", label: "Credit Amount (DM)", type: "number" as const, category: "Loan Details" },
  { key: "duration_months", label: "Duration (months)", type: "number" as const, category: "Loan Details" },
  { key: "installment_rate", label: "Installment Rate (%)", type: "number" as const, category: "Loan Details" },
  { key: "savings_bonds", label: "Savings / Bonds", type: "select" as const, category: "Applicant Profile" },
  { key: "employment_since", label: "Employment Duration", type: "select" as const, category: "Applicant Profile" },
  { key: "personal_status_sex", label: "Personal Status / Sex", type: "select" as const, category: "Applicant Profile" },
  { key: "other_debtors", label: "Other Debtors", type: "select" as const, category: "Applicant Profile" },
  { key: "present_residence", label: "Years at Residence", type: "number" as const, category: "Applicant Profile" },
  { key: "property", label: "Property", type: "select" as const, category: "Financial Status" },
  { key: "age_years", label: "Age", type: "number" as const, category: "Applicant Profile" },
  { key: "other_installment", label: "Other Installments", type: "select" as const, category: "Financial Status" },
  { key: "housing", label: "Housing", type: "select" as const, category: "Applicant Profile" },
  { key: "existing_credits", label: "Existing Credits", type: "number" as const, category: "Financial Status" },
  { key: "job", label: "Job Category", type: "select" as const, category: "Applicant Profile" },
  { key: "dependents", label: "Dependents", type: "number" as const, category: "Applicant Profile" },
  { key: "telephone", label: "Telephone", type: "select" as const, category: "Applicant Profile" },
  { key: "foreign_worker", label: "Foreign Worker", type: "select" as const, category: "Applicant Profile" },
];

const DEFAULT_FORM: Record<string, string | number> = {
  status_checking: "A11 (< 0 DM)",
  duration_months: 24,
  credit_history: "A32 (existing credits paid back)",
  purpose: "A42 (furniture)",
  credit_amount: 2500,
  savings_bonds: "A65 (unknown/none)",
  employment_since: "A73 (1–4 years)",
  installment_rate: 4,
  personal_status_sex: "A93 (male: single)",
  other_debtors: "A101 (none)",
  present_residence: 3,
  property: "A122 (building society / life insurance)",
  age_years: 30,
  other_installment: "A143 (none)",
  housing: "A152 (own)",
  existing_credits: 1,
  job: "A173 (skilled/self-employed)",
  dependents: 1,
  telephone: "A191 (none)",
  foreign_worker: "A201 (yes)",
};

function extractCode(value: string): string {
  const match = value.match(/^([A-Z]\d+)/);
  return match ? match[1] : value;
}

function PredictorForm() {
  const { loading, error, result, submit, reset } = usePredictor();
  const [form, setForm] = useState<Record<string, string | number>>({ ...DEFAULT_FORM });

  function handleChange(key: string, value: string | number) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const payload: Record<string, string | number> = {};
    for (const field of ALL_FIELDS) {
      const val = form[field.key];
      if (field.type === "select") {
        payload[field.key] = extractCode(String(val));
      } else {
        payload[field.key] = Number(val);
      }
    }
    submit(payload as any);
  }

  function handleReset() {
    setForm({ ...DEFAULT_FORM });
    reset();
  }

  const categories = [...new Set(ALL_FIELDS.map((f) => f.category))];

  return (
    <div className="space-y-8">
      <form onSubmit={handleSubmit} className="space-y-10">
        {categories.map((cat) => (
          <div key={cat}>
            <h3 className="text-xs font-mono tracking-widest text-neon/60 uppercase mb-4">{cat}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {ALL_FIELDS.filter((f) => f.category === cat).map((field) => (
                <div key={field.key}>
                  {field.type === "select" ? (
                    <div className="relative">
                      <div
                        className={clsx(
                          "relative rounded-xl border transition-all duration-300",
                          "border-white/[0.06] bg-white/[0.03] hover:border-white/[0.12]"
                        )}
                      >
                        <select
                          value={String(form[field.key] || "")}
                          onChange={(e) => handleChange(field.key, e.target.value)}
                          className="w-full bg-transparent px-4 pt-5 pb-2 text-sm text-white appearance-none focus:outline-none"
                        >
                          {(CATEGORICAL_OPTIONS[field.key] || []).map((opt) => (
                            <option key={opt} value={opt} className="bg-surface text-white">
                              {opt}
                            </option>
                          ))}
                        </select>
                        <label className="absolute left-4 top-1.5 text-[10px] text-neon/70 font-medium tracking-wider uppercase pointer-events-none">
                          {field.label}
                        </label>
                        <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none">
                          <svg className="w-3.5 h-3.5 text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <Input
                      id={field.key}
                      label={field.label}
                      type="number"
                      value={form[field.key]}
                      onChange={(e) => handleChange(field.key, e.target.value)}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}

        <div className="flex gap-3 pt-2">
          <Button type="submit" loading={loading} size="lg" className="flex-1">
            {loading ? "Analyzing..." : "Assess Credit Risk"}
            {!loading && <ArrowRight className="w-4 h-4" />}
          </Button>
          <Button type="button" variant="outline" size="lg" onClick={handleReset}>
            Reset
          </Button>
        </div>
      </form>

      <AnimatePresence mode="wait">
        {loading && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass rounded-2xl p-8 text-center"
          >
            <Loader2 className="w-8 h-8 animate-spin text-neon mx-auto mb-3" />
            <p className="text-sm text-white/60">Running assessment...</p>
          </motion.div>
        )}

        {error && (
          <motion.div
            key="error"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <Card className="border-red-500/20">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
                <div>
                  <p className="text-sm font-medium text-red-400">Assessment Failed</p>
                  <p className="text-xs text-red-400/60 mt-1">{error}</p>
                </div>
              </div>
            </Card>
          </motion.div>
        )}

        {result && (
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            <Card glow className={result.prediction_label === 1 ? "border-red-500/20" : "border-neon/20"}>
              <div className="flex items-start gap-4">
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
                    result.prediction_label === 1
                      ? "bg-red-500/10 text-red-400"
                      : "bg-neon/10 text-neon"
                  }`}
                >
                  {result.prediction_label === 1 ? (
                    <AlertTriangle className="w-6 h-6" />
                  ) : (
                    <CheckCircle2 className="w-6 h-6" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-lg font-bold text-white">{result.prediction}</p>
                  <p className="text-xs text-white/40 mt-1">{result.explanation}</p>
                </div>
              </div>
            </Card>

            <div className="grid grid-cols-3 gap-3">
              <Card className="text-center p-4">
                <Shield className="w-4 h-4 text-neon mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">
                  {(result.risk_probability * 100).toFixed(1)}%
                </p>
                <p className="text-[10px] text-white/30 mt-1">Risk Probability</p>
              </Card>
              <Card className="text-center p-4">
                <TrendingUp className="w-4 h-4 text-emerald-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{result.threshold.toFixed(2)}</p>
                <p className="text-[10px] text-white/30 mt-1">Decision Threshold</p>
              </Card>
              <Card className="text-center p-4">
                <Activity className="w-4 h-4 text-blue-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{result.risk_score.toFixed(4)}</p>
                <p className="text-[10px] text-white/30 mt-1">Raw Score</p>
              </Card>
            </div>

            <Card className="p-4">
              <div className="flex items-start gap-3">
                <Info className="w-4 h-4 text-neon/60 mt-0.5 shrink-0" />
                <div className="text-xs text-white/40 leading-relaxed">
                  This assessment uses a strict 0.49 decision threshold optimized for a 5:1 asymmetric cost
                  model — misclassifying a high-risk applicant costs 5&times; more than rejecting a safe one.
                  The model prioritizes capital protection over approval volume.
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

import { clsx } from "clsx";

export { PredictorForm };
