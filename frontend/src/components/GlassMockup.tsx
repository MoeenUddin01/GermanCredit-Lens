"use client";

import { useRef } from "react";
import { motion, useMotionValue, useTransform } from "framer-motion";
import { Shield, TrendingUp, Activity } from "lucide-react";

function GlassMockup() {
  const ref = useRef<HTMLDivElement>(null);

  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const rotateX = useTransform(y, [-0.5, 0.5], [8, -8]);
  const rotateY = useTransform(x, [-0.5, 0.5], [-8, 8]);

  const glareX = useTransform(x, [-0.5, 0.5], [0, 100]);
  const glareY = useTransform(y, [-0.5, 0.5], [0, 100]);

  function handlePointerMove(e: React.PointerEvent) {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    x.set(px);
    y.set(py);
  }

  function handlePointerLeave() {
    x.set(0);
    y.set(0);
  }

  const stats = [
    { label: "Risk Threshold", value: "0.49", icon: Shield, color: "text-neon" },
    { label: "ROC-AUC", value: "0.76", icon: Activity, color: "text-blue-400" },
    { label: "Recall", value: "76.7%", icon: TrendingUp, color: "text-emerald-400" },
  ];

  return (
    <div className="perspective-1000 w-full max-w-sm mx-auto">
      <motion.div
        ref={ref}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        style={{ rotateX, rotateY }}
        className="relative preserve-3d cursor-pointer"
      >
        <div className="relative glass-strong rounded-3xl p-1 overflow-hidden">
          <motion.div
            className="pointer-events-none absolute inset-0 rounded-3xl opacity-30"
            style={{
              background: `radial-gradient(circle at ${glareX}% ${glareY}%, rgba(204,255,0,0.15) 0%, transparent 60%)`,
            }}
          />

          <div className="relative rounded-2xl bg-gradient-to-b from-surface-light to-surface p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-white/40 font-mono tracking-widest uppercase">pulse.</p>
                <p className="text-[10px] text-white/20 font-mono mt-0.5">Credit Risk Engine</p>
              </div>
              <div className="w-8 h-8 rounded-full bg-neon/10 border border-neon/20 flex items-center justify-center">
                <Shield className="w-4 h-4 text-neon" />
              </div>
            </div>

            <div className="text-center">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2, type: "spring" }}
              >
                <p className="text-5xl font-bold text-gradient tracking-tight">pulse.</p>
              </motion.div>
              <p className="mt-3 text-xs text-white/30 font-mono">
                Real-time Credit Assessment
              </p>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {stats.map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + i * 0.1 }}
                  className="glass rounded-xl p-3 text-center"
                >
                  <stat.icon className={`w-3.5 h-3.5 mx-auto mb-1.5 ${stat.color}`} />
                  <p className="text-sm font-bold text-white">{stat.value}</p>
                  <p className="text-[9px] text-white/30 mt-0.5 truncate">{stat.label}</p>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="glass rounded-xl px-4 py-3 flex items-center justify-between"
            >
              <span className="text-[10px] text-white/30 font-mono">Status</span>
              <span className="text-[10px] text-neon font-mono tracking-wider">● Operational</span>
            </motion.div>
          </div>
        </div>

        <motion.div
          className="absolute -inset-4 rounded-3xl bg-neon/5 blur-3xl -z-10"
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.div>
    </div>
  );
}

export { GlassMockup };
