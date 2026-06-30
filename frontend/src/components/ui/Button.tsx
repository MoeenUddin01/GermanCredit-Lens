"use client";

import { forwardRef } from "react";
import { clsx } from "clsx";
import { Loader2 } from "lucide-react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "outline";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  children: React.ReactNode;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", size = "md", loading, className, children, disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={clsx(
          "relative inline-flex items-center justify-center font-medium transition-all duration-300 rounded-xl",
          "focus:outline-none focus:ring-2 focus:ring-neon/40 focus:ring-offset-2 focus:ring-offset-surface",
          "disabled:opacity-40 disabled:cursor-not-allowed",
          {
            primary:
              "bg-neon text-black hover:bg-neon-hover active:scale-[0.97] neon-glow",
            ghost:
              "glass hover:bg-white/[0.1] active:scale-[0.97] text-white/80 hover:text-white",
            outline:
              "border border-white/10 hover:border-neon/50 text-white/70 hover:text-neon bg-transparent",
          }[variant],
          {
            sm: "h-9 px-4 text-sm gap-2",
            md: "h-11 px-6 text-sm gap-2.5",
            lg: "h-13 px-8 text-base gap-3",
          }[size],
          className
        )}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";

export { Button };
export type { ButtonProps };
