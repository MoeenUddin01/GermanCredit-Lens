"use client";

import { forwardRef, useState } from "react";
import { clsx } from "clsx";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, className, id, value, onChange, onFocus, onBlur, ...props }, ref) => {
    const [focused, setFocused] = useState(false);
    const hasValue = value !== undefined && value !== null && String(value).length > 0;
    const isFloating = focused || hasValue;

    return (
      <div className="relative">
        <div
          className={clsx(
            "relative rounded-xl border transition-all duration-300",
            error
              ? "border-red-500/50 bg-red-500/5"
              : focused
              ? "border-neon/50 bg-white/[0.06]"
              : "border-white/[0.06] bg-white/[0.03] hover:border-white/[0.12]"
          )}
        >
          <input
            ref={ref}
            id={id}
            value={value}
            onChange={onChange}
            onFocus={(e) => {
              setFocused(true);
              onFocus?.(e);
            }}
            onBlur={(e) => {
              setFocused(false);
              onBlur?.(e);
            }}
            className={clsx(
              "w-full bg-transparent px-4 pt-5 pb-2 text-sm text-white placeholder-transparent",
              "focus:outline-none focus:ring-0",
              "autofill:bg-transparent",
              className
            )}
            placeholder={label}
            {...props}
          />
          <label
            htmlFor={id}
            className={clsx(
              "absolute left-4 transition-all duration-200 pointer-events-none select-none",
              isFloating
                ? "top-1.5 text-[10px] text-neon/70 font-medium tracking-wider uppercase"
                : "top-1/2 -translate-y-1/2 text-sm text-white/40"
            )}
          >
            {label}
          </label>
          {error && (
            <p className="px-4 pb-1.5 pt-1 text-[11px] text-red-400">{error}</p>
          )}
        </div>
      </div>
    );
  }
);
Input.displayName = "Input";

export { Input };
export type { InputProps };
