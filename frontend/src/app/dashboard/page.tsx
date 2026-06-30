"use client";

import { motion } from "framer-motion";
import { ArrowLeft, Shield } from "lucide-react";
import Link from "next/link";
import { PredictorForm } from "@/components/PredictorForm";
import { Button } from "@/components/ui/Button";

export default function DashboardPage() {
  return (
    <main className="relative min-h-screen">
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(204,255,0,0.02),transparent_50%)] pointer-events-none" />

      <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <header className="flex items-center justify-between mb-10">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="w-4 h-4" />
                Back
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-neon" />
              <span className="text-sm font-bold tracking-tight text-white">pulse.</span>
            </div>
          </div>
          <div className="text-[10px] font-mono text-white/20 tracking-wider">
            Credit Risk Engine v1.0
          </div>
        </header>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="mb-8">
            <p className="text-xs font-mono tracking-widest text-neon/60 uppercase">
              Credit Assessment
            </p>
            <h1 className="text-3xl font-bold text-gradient mt-2">
              New Application Review
            </h1>
            <p className="text-sm text-white/40 mt-2 max-w-xl">
              Enter the applicant&apos;s financial details below. The engine will evaluate
              creditworthiness using the asymmetric cost model and return an instant decision.
            </p>
          </div>

          <div className="glass rounded-3xl p-6 sm:p-8 lg:p-10">
            <PredictorForm />
          </div>
        </motion.div>
      </div>
    </main>
  );
}
