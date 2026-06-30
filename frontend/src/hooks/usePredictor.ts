"use client";

import { useState, useCallback } from "react";
import type { CreditApplication, PredictionResponse } from "@/services/api";
import { predictCreditRisk } from "@/services/api";

export interface PredictorState {
  loading: boolean;
  error: string | null;
  result: PredictionResponse | null;
}

export function usePredictor() {
  const [state, setState] = useState<PredictorState>({
    loading: false,
    error: null,
    result: null,
  });

  const submit = useCallback(async (data: CreditApplication) => {
    setState({ loading: true, error: null, result: null });
    try {
      const result = await predictCreditRisk(data);
      setState({ loading: false, error: null, result });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Prediction failed";
      setState({ loading: false, error: message, result: null });
    }
  }, []);

  const reset = useCallback(() => {
    setState({ loading: false, error: null, result: null });
  }, []);

  return { ...state, submit, reset };
}
