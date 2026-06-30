"use client";

import { motion } from "framer-motion";
import { ArrowRight, Shield, TrendingUp, Activity, BarChart3 } from "lucide-react";
import Link from "next/link";
import { GlassMockup } from "@/components/GlassMockup";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

const metrics = [
  { icon: Shield, label: "Asymmetric Cost", value: "5:1", desc: "Bad → Good costs 5x more" },
  { icon: TrendingUp, label: "Test Recall", value: "76.7%", desc: "Bad risk detection rate" },
  { icon: Activity, label: "ROC-AUC", value: "0.76", desc: "Model discrimination power" },
  { icon: BarChart3, label: "Threshold", value: "0.49", desc: "Defensive decision boundary" },
];

export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(204,255,0,0.03),transparent_50%),radial-gradient(ellipse_at_bottom_left,rgba(59,130,246,0.02),transparent_50%)] pointer-events-none" />
      <div className="fixed top-[-20%] right-[-10%] w-[40%] h-[40%] bg-neon/5 rounded-full blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-20%] left-[-10%] w-[40%] h-[40%] bg-blue-500/5 rounded-full blur-[120px] pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <header className="flex items-center justify-between py-6">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-neon" />
            <span className="text-sm font-bold tracking-tight text-white">pulse.</span>
          </div>
          <div className="flex items-center gap-3">
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">
                Dashboard
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button size="sm">
                Launch App
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
          </div>
        </header>

        <section className="py-20 lg:py-32">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="inline-flex items-center gap-2 glass rounded-full px-4 py-1.5 mb-6">
                <span className="w-1.5 h-1.5 rounded-full bg-neon animate-pulse" />
                <span className="text-[10px] font-mono text-white/40 tracking-wider uppercase">
                  Production Ready &mdash; v1.0
                </span>
              </div>

              <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-gradient leading-[1.1]">
                Credit Risk,
                <br />
                <span className="text-neon">Decoded</span>
              </h1>

              <p className="mt-6 text-base text-white/40 leading-relaxed max-w-lg">
                Real-time creditworthiness assessment powered by XGBoost. 
                Our engine uses a strict asymmetric cost model to prioritize 
                capital protection over approval volume.
              </p>

              <div className="flex gap-3 mt-8">
                <Link href="/dashboard">
                  <Button size="lg">
                    Start Assessment
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                </Link>
                <Link href="/dashboard">
                  <Button variant="outline" size="lg">
                    Learn More
                  </Button>
                </Link>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-12">
                {metrics.map((m) => (
                  <div key={m.label} className="glass rounded-xl p-3">
                    <m.icon className="w-3.5 h-3.5 text-neon mb-2" />
                    <p className="text-lg font-bold text-white">{m.value}</p>
                    <p className="text-[10px] text-white/30 mt-0.5">{m.label}</p>
                    <p className="text-[9px] text-white/20 mt-0.5">{m.desc}</p>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="hidden lg:block"
            >
              <GlassMockup />
            </motion.div>
          </div>
        </section>

        <section className="pb-20">
          <div className="text-center mb-12">
            <p className="text-xs font-mono tracking-widest text-neon/60 uppercase">How It Works</p>
            <h2 className="text-2xl font-bold text-white mt-2">Three-Step Assessment</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              { step: "01", title: "Submit Application", desc: "Enter applicant financial data through our secure form interface." },
              { step: "02", title: "XGBoost Analysis", desc: "Our engine evaluates risk using a model optimized for a 5:1 asymmetric cost matrix." },
              { step: "03", title: "Instant Verdict", desc: "Receive an approved/rejected decision with calibrated risk probability and explanation." },
            ].map((item) => (
              <Card key={item.step} hover className="text-center p-8">
                <p className="text-3xl font-bold text-neon/30 font-mono">{item.step}</p>
                <h3 className="text-sm font-bold text-white mt-4">{item.title}</h3>
                <p className="text-xs text-white/40 mt-2 leading-relaxed">{item.desc}</p>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
