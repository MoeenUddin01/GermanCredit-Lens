export interface CreditApplication {
  status_checking: string;
  duration_months: number;
  credit_history: string;
  purpose: string;
  credit_amount: number;
  savings_bonds: string;
  employment_since: string;
  installment_rate: number;
  personal_status_sex: string;
  other_debtors: string;
  present_residence: number;
  property: string;
  age_years: number;
  other_installment: string;
  housing: string;
  existing_credits: number;
  job: string;
  dependents: number;
  telephone: string;
  foreign_worker: string;
}

export interface PredictionResponse {
  risk_score: number;
  risk_probability: number;
  prediction: string;
  prediction_label: number;
  threshold: number;
  explanation: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function predictCreditRisk(
  data: CreditApplication
): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Prediction failed" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
