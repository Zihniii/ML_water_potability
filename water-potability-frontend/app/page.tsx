"use client";

import { useState } from "react";
import axios from "axios";

const fieldMeta = [
  { key: "ph",              label: "pH",              icon: "⚗️", unit: "0–14 scale",   placeholder: "e.g. 7.2"   },
  { key: "Hardness",        label: "Hardness",        icon: "💎", unit: "mg/L",         placeholder: "e.g. 204"   },
  { key: "Solids",          label: "Solids (TDS)",    icon: "🪨", unit: "ppm",          placeholder: "e.g. 20791" },
  { key: "Chloramines",     label: "Chloramines",     icon: "🧪", unit: "ppm",          placeholder: "e.g. 7.3"   },
  { key: "Sulfate",         label: "Sulfate",         icon: "⚛️", unit: "mg/L",         placeholder: "e.g. 368"   },
  { key: "Conductivity",    label: "Conductivity",    icon: "⚡", unit: "μS/cm",        placeholder: "e.g. 564"   },
  { key: "Organic_carbon",  label: "Organic carbon",  icon: "🌿", unit: "ppm",          placeholder: "e.g. 10.4"  },
  { key: "Trihalomethanes", label: "Trihalomethanes", icon: "🔬", unit: "μg/L",         placeholder: "e.g. 86"    },
  { key: "Turbidity",       label: "Turbidity",       icon: "🌊", unit: "NTU",          placeholder: "e.g. 3.9"   },
];

type FormData = {
  ph: string; Hardness: string; Solids: string; Chloramines: string;
  Sulfate: string; Conductivity: string; Organic_carbon: string;
  Trihalomethanes: string; Turbidity: string;
};

const emptyForm: FormData = {
  ph: "", Hardness: "", Solids: "", Chloramines: "",
  Sulfate: "", Conductivity: "", Organic_carbon: "",
  Trihalomethanes: "", Turbidity: "",
};

export default function Home() {
  const [formData, setFormData] = useState<FormData>(emptyForm);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handlePredict = async () => {
    setError("");
    setPrediction(null);

    for (const f of fieldMeta) {
      const val = parseFloat(formData[f.key as keyof FormData]);
      if (isNaN(val)) {
        setError(`Please enter a valid number for "${f.label}".`);
        return;
      }
    }

    try {
      setLoading(true);
      const payload = Object.fromEntries(
        fieldMeta.map((f) => [f.key, parseFloat(formData[f.key as keyof FormData])])
      );
      const response = await axios.post("http://localhost:8000/predict", payload);
      setPrediction(response.data.Potability);
    } catch {
      setError("Could not reach the FastAPI backend. Make sure it's running on localhost:8000.");
    } finally {
      setLoading(false);
    }
  };

  const ph    = parseFloat(formData.ph);
  const turb  = parseFloat(formData.Turbidity);
  const tds   = parseFloat(formData.Solids);
  const phOk  = ph >= 6.5 && ph <= 8.5;
  const turbOk = turb <= 4;
  const tdsOk  = tds <= 500;

  return (
    <main className="min-h-screen bg-[#0a0f1e] text-white flex items-center justify-center p-6">

      {/* Ambient water glow blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] rounded-full bg-[#1D9E75]/10 blur-[120px]" />
        <div className="absolute bottom-[-5%] right-[15%] w-[400px] h-[400px] rounded-full bg-[#0e7490]/10 blur-[100px]" />
        <div className="absolute top-[40%] left-[-5%] w-[300px] h-[300px] rounded-full bg-[#5DCAA5]/5 blur-[80px]" />
      </div>

      <div className="relative w-full max-w-3xl">

        {/* Header */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center mb-5">
            <div className="relative">
              {/* Ripple rings */}
              <span className="absolute inset-0 rounded-full border border-[#1D9E75]/40 animate-ping" />
              <span className="absolute -inset-2 rounded-full border border-[#1D9E75]/20 animate-ping [animation-delay:0.4s]" />
              <div className="relative w-16 h-16 rounded-full bg-[#1D9E75]/10 border border-[#1D9E75]/30 flex items-center justify-center">
                <WaterDropSVG />
              </div>
            </div>
          </div>
          <h1 className="text-4xl font-bold tracking-tight mb-2">
            Water Potability
            <span className="text-[#5DCAA5]"> Predictor</span>
          </h1>
          <p className="text-white/50 text-sm">
            Enter water quality parameters to check if it's safe to drink.
          </p>
        </div>

        {/* Form grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mb-6">
          {fieldMeta.map((f) => (
            <div
              key={f.key}
              className="group rounded-2xl bg-white/[0.04] border border-white/[0.08] p-4
                         hover:border-[#1D9E75]/40 focus-within:border-[#1D9E75]/60
                         transition-colors duration-200"
            >
              <label className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-white/40 mb-2">
                <span>{f.icon}</span>
                {f.label}
              </label>
              <input
                type="number"
                step="any"
                name={f.key}
                value={formData[f.key as keyof FormData]}
                onChange={handleChange}
                placeholder={f.placeholder}
                className="w-full bg-transparent text-white text-base font-medium
                           outline-none placeholder:text-white/20"
              />
              <p className="text-[11px] text-white/25 mt-1">{f.unit}</p>
            </div>
          ))}
        </div>

        {/* Submit */}
        <button
          onClick={handlePredict}
          disabled={loading}
          className="w-full py-4 rounded-2xl font-semibold text-base transition-all duration-200
                     bg-[#1D9E75] hover:bg-[#5DCAA5] hover:text-[#0a0f1e]
                     disabled:opacity-40 disabled:cursor-not-allowed
                     flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <SpinnerSVG />
              Analyzing sample…
            </>
          ) : (
            <>
              <WaveIconSVG />
              Analyze water sample
            </>
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-xl bg-red-500/10 border border-red-500/20 p-4 text-red-300 text-sm flex items-start gap-2">
            <span className="mt-0.5">⚠️</span>
            {error}
          </div>
        )}

        {/* Result */}
        {prediction !== null && (
          <div className="mt-6 rounded-2xl border border-white/10 overflow-hidden">
            {/* Animated wave bar */}
            <div
              className={`h-1 w-full ${prediction === 1 ? "bg-gradient-to-r from-[#1D9E75] via-[#9FE1CB] to-[#1D9E75]" : "bg-gradient-to-r from-[#D85A30] via-[#F5C4B3] to-[#D85A30]"}`}
              style={{ backgroundSize: "200%", animation: "waveflow 2s linear infinite" }}
            />

            <div className="bg-white/[0.03] p-8 text-center">
              {prediction === 1 ? (
                <>
                  <div className="w-20 h-20 rounded-full bg-[#1D9E75]/10 border border-[#1D9E75]/30 flex items-center justify-center mx-auto mb-4">
                    <span className="text-4xl">💧</span>
                  </div>
                  <h2 className="text-2xl font-bold text-[#5DCAA5] mb-1">Water is Potable</h2>
                  <p className="text-white/40 text-sm">
                    This water meets safety thresholds and appears safe for drinking.
                  </p>
                </>
              ) : (
                <>
                  <div className="w-20 h-20 rounded-full bg-[#D85A30]/10 border border-[#D85A30]/30 flex items-center justify-center mx-auto mb-4">
                    <span className="text-4xl">⚠️</span>
                  </div>
                  <h2 className="text-2xl font-bold text-[#F0997B] mb-1">Water is Not Potable</h2>
                  <p className="text-white/40 text-sm">
                    This water may contain harmful contaminant levels. Do not drink.
                  </p>
                </>
              )}

              {/* Quick stats */}
              <div className="grid grid-cols-3 gap-3 mt-6 pt-6 border-t border-white/10">
                <StatPill label="pH" value={isNaN(ph) ? "—" : ph.toFixed(1)} ok={phOk} hasValue={!isNaN(ph)} />
                <StatPill label="Turbidity" value={isNaN(turb) ? "—" : `${turb.toFixed(1)} NTU`} ok={turbOk} hasValue={!isNaN(turb)} />
                <StatPill label="TDS" value={isNaN(tds) ? "—" : `${Math.round(tds).toLocaleString()} ppm`} ok={tdsOk} hasValue={!isNaN(tds)} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Waveflow keyframe injected inline */}
      <style>{`
        @keyframes waveflow {
          0%   { background-position: 0% 0%; }
          100% { background-position: 200% 0%; }
        }
      `}</style>
    </main>
  );
}

function StatPill({ label, value, ok, hasValue }: { label: string; value: string; ok: boolean; hasValue: boolean }) {
  return (
    <div className="rounded-xl bg-white/[0.04] border border-white/[0.06] p-3">
      <p className="text-[10px] uppercase tracking-widest text-white/30 mb-1">{label}</p>
      <p className={`text-sm font-semibold ${!hasValue ? "text-white/30" : ok ? "text-[#5DCAA5]" : "text-[#F0997B]"}`}>
        {value}
      </p>
    </div>
  );
}

function WaterDropSVG() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <path
        d="M14 3C14 3 7 11.5 7 17C7 20.866 10.134 24 14 24C17.866 24 21 20.866 21 17C21 11.5 14 3 14 3Z"
        fill="#1D9E75" fillOpacity="0.3" stroke="#5DCAA5" strokeWidth="1.5" strokeLinejoin="round"
      />
      <path d="M10.5 18.5C11 20.5 12.5 21.5 14 21.5" stroke="#9FE1CB" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function WaveIconSVG() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <path d="M2 9C2 9 4 6 6 9C8 12 10 6 12 9C14 12 16 9 16 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SpinnerSVG() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style={{ animation: "spin 0.8s linear infinite" }}>
      <circle cx="9" cy="9" r="7" stroke="currentColor" strokeOpacity="0.3" strokeWidth="2" />
      <path d="M9 2a7 7 0 0 1 7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </svg>
  );
}