"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Save, CheckCircle2, Loader2, Terminal, Info } from "lucide-react";
import { useSettings, useSaveSettings } from "@/hooks/use-settings";

export default function SettingsPage() {
  const { pythonBin: savedPythonBin, loading, refetch } = useSettings();
  const { save, saving } = useSaveSettings();
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Hydrate the form once the stored value loads, and re-apply the canonical
  // value after each save. React-sanctioned "adjusting state during render"
  // (compare against the previous saved value rather than mutating an effect).
  const [prevSaved, setPrevSaved] = useState(savedPythonBin);
  if (savedPythonBin !== prevSaved) {
    setPrevSaved(savedPythonBin);
    setValue(savedPythonBin);
  }

  const handleSave = async () => {
    setError(null);
    setSaved(false);
    const ok = await save(value.trim());
    if (ok) {
      setSaved(true);
      refetch();
    } else {
      setError("Failed to save. Please try again.");
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Global runtime preferences for training jobs
        </p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-xl border border-border/50 overflow-hidden"
      >
        <div className="px-5 py-3 text-sm font-semibold flex items-center gap-2">
          <Terminal className="w-4 h-4 text-muted-foreground" />
          Python Interpreter
        </div>
        <div className="px-5 pb-5">
          <label className="block text-xs font-medium mb-1.5">
            Interpreter path
          </label>
          <input
            type="text"
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setSaved(false);
            }}
            placeholder="/home/user/.venv/bin/python"
            spellCheck={false}
            className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm font-mono"
          />
          <div className="flex items-start gap-2 mt-2">
            <Info className="w-3.5 h-3.5 text-muted-foreground shrink-0 mt-0.5" />
            <p className="text-xs text-muted-foreground leading-relaxed">
              Training scripts run with this interpreter. When it points into a
              virtualenv bin directory, the <span className="font-mono">torchrun</span> and{" "}
              <span className="font-mono">deepspeed</span> launchers resolve from the same
              directory, so everything uses that venv. Leave blank to use{" "}
              <span className="font-mono">python3</span> from PATH.
            </p>
          </div>

          <div className="flex items-center gap-3 mt-4">
            <button
              onClick={handleSave}
              disabled={saving}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50 glow-primary"
            >
              {saving ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4" />
                  Save
                </>
              )}
            </button>
            {saved && (
              <span className="inline-flex items-center gap-1.5 text-xs text-green-400">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Saved
              </span>
            )}
            {error && <span className="text-xs text-red-400">{error}</span>}
          </div>

          <div className="mt-4 pt-4 border-t border-border/50">
            <p className="text-xs font-medium text-muted-foreground mb-1">
              Currently active
            </p>
            <p className="text-sm font-mono">
              {loading
                ? "…"
                : savedPythonBin
                ? savedPythonBin
                : "python3 (system default)"}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
