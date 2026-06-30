"use client";

import { clsx } from "clsx";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  glow?: boolean;
  hover?: boolean;
}

function Card({ children, className, glow, hover }: CardProps) {
  return (
    <div
      className={clsx(
        "glass rounded-2xl p-6",
        glow && "neon-glow",
        hover && "hover:bg-white/[0.08] hover:border-white/[0.12] hover:-translate-y-0.5",
        "transition-all duration-300",
        className
      )}
    >
      {children}
    </div>
  );
}

export { Card };
export type { CardProps };
