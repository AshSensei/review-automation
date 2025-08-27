import React, { useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useNavigate } from "react-router-dom";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

import { Lightbulb, Trophy, Shield, Star, AlertTriangle } from "lucide-react";
import { FadeTransition } from "@/components/ui/FadeTransition";


// --- TYPESCRIPT INTERFACES ---
// Fixed to match the actual backend JSON structure

interface Issue {
  issue_name: string;
  description: string;
  frequency: number;
  severity: string;
  example_quote: string;
  type?: string;
}

interface Theme {
  confidence: number;
  overall_sentiment: "positive" | "negative" | "mixed";
  positive_points: string[]; // FIX: Renamed from 'positives'
  negative_points: string[]; // FIX: Renamed from 'negatives'
  representative_quote: string; // FIX: Renamed from 'example_quote'
}

interface Metrics {
  total_reviews: number;
  average_rating: number;
  rating_distribution: Record<string, number>;
  sentiment_distribution: Record<string, number>;
  average_review_length: number;
  analysis_quality: string;
}

interface Recommendation {
  recommendation: string;
  priority: string; // FIX: Changed from union type to 'string' to allow for "High", "Medium-High", etc.
  impact: string;
  rationale?: string;
}

interface Insights {
  executive_summary: string;
  recommendations: Recommendation[];
  key_insights: string[];
}

interface ProductAnalysis {
  themes: Record<string, Theme>;
  issues: Issue[];
  metrics: Metrics;
  insights: Insights;
  analysis_metadata: {
    total_reviews: number;
    analysis_date: string;
    product_type: string;
    model_used: string;
    analysis_time_seconds: number;
    token_usage?: Record<string, number>;
  };
}

interface StrategicInsights {
  competitive_advantages: string[];
  areas_to_improve: string[];
  recommendations: Recommendation[];
}

interface ComparisonData {
  shared_themes: string[];
  unique_to_product_a: string[];
  unique_to_product_b: string[];
  theme_sentiment_comparison: Record<
    string,
    { product_a: string; product_b: string }
  >;
  summary_table: string;
  strategic_insights: StrategicInsights;
}

export interface FullApiResponse {
  product_a?: ProductAnalysis;
  product_b?: ProductAnalysis;
  comparison?: ComparisonData;
}

interface StatusUpdate {
  message: string;
  timestamp: number;
}
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000/";
// Theme reconciliation helper - maps similar themes to common names
const reconcileThemes = (
  productAThemes: string[],
  productBThemes: string[]
) => {
  const themeMapping: Record<string, string[]> = {
    Performance: [
      "Performance and Input Response",
      "Performance and Responsiveness",
      "Performance",
      "Input Response",
      "Responsiveness",
    ],
    "Battery Life": [
      "Battery Life and Charging Dock",
      "Battery and Power Management",
      "Battery Life",
      "Battery",
      "Power Management",
      "Charging",
    ],
    "Build Quality": [
      "Build Quality and QA",
      "Build Quality, Accessories, and Price",
      "Build Quality",
      "Quality",
      "Construction",
    ],
    Connectivity: [
      "Connectivity and Software/App",
      "Connectivity and Compatibility",
      "Connectivity",
      "Connection",
      "Wireless",
      "Pairing",
    ],
    Thumbsticks: [
      "Thumbsticks and Anti-Friction Rings",
      "Thumbsticks (Hall Effect), Deadzone, and Calibration",
      "Thumbsticks",
      "Sticks",
      "Joysticks",
      "Hall Effect",
    ],
    Ergonomics: [
      "Ergonomics and Button Layout",
      "Ergonomics and Comfort",
      "Ergonomics",
      "Comfort",
      "Feel",
    ],
    Customization: [
      "Customization and On-Device Screen",
      "Customization and Software",
      "Customization",
      "Software",
      "Profiles",
    ],
    "Back Buttons": [
      "Back Buttons and Controls",
      "Back Buttons",
      "Paddles",
      "Controls",
    ],
  };

  const sharedThemes: {
    canonical: string;
    productA: string;
    productB: string;
  }[] = [];
  const uniqueToA: string[] = [];
  const uniqueToB: string[] = [];

  const findCanonicalTheme = (themeName: string): string | null => {
    for (const [canonical, variants] of Object.entries(themeMapping)) {
      if (
        variants.some(
          (variant) =>
            themeName.toLowerCase().includes(variant.toLowerCase()) ||
            variant.toLowerCase().includes(themeName.toLowerCase())
        )
      ) {
        return canonical;
      }
    }
    return null;
  };

  const processedA = new Set<string>();
  const processedB = new Set<string>();

  // Find shared themes
  for (const themeA of productAThemes) {
    if (processedA.has(themeA)) continue;

    const canonicalA = findCanonicalTheme(themeA);
    if (!canonicalA) continue;

    for (const themeB of productBThemes) {
      if (processedB.has(themeB)) continue;

      const canonicalB = findCanonicalTheme(themeB);
      if (canonicalA === canonicalB) {
        sharedThemes.push({
          canonical: canonicalA,
          productA: themeA,
          productB: themeB,
        });
        processedA.add(themeA);
        processedB.add(themeB);
        break;
      }
    }
  }

  // Add remaining themes as unique
  for (const themeA of productAThemes) {
    if (!processedA.has(themeA)) {
      uniqueToA.push(themeA);
    }
  }

  for (const themeB of productBThemes) {
    if (!processedB.has(themeB)) {
      uniqueToB.push(themeB);
    }
  }

  return { sharedThemes, uniqueToA, uniqueToB };
};

const formatThemeName = (name: string = ""): string => {
  return name
    .split(/[_\s]+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const capitalize = (s: string) => {
  if (!s) return "";
  return s.charAt(0).toUpperCase() + s.slice(1);
};

const getSentimentClass = (
  sentiment: string = "neutral",
  element: "text" | "bg" | "border" = "text"
) => {
  const sentimentLower = sentiment.toLowerCase();
  const styles: {
    [key: string]: { text: string; bg: string; border: string };
  } = {
    positive: {
      text: "text-green-700",
      bg: "bg-green-50",
      border: "border-green-300",
    },
    negative: {
      text: "text-red-700",
      bg: "bg-red-50",
      border: "border-red-300",
    },
    neutral: {
      text: "text-gray-700",
      bg: "bg-gray-50",
      border: "border-gray-300",
    },
  };
  return styles[sentimentLower]?.[element] || styles.neutral[element];
};

const getPriorityClass = (priority: string = "low") => {
  switch (priority.toLowerCase()) {
    case "high":
      return "bg-red-100 text-red-800 border-red-200";
    case "medium":
      return "bg-yellow-100 text-yellow-800 border-yellow-200";
    case "low":
      return "bg-green-100 text-green-800 border-green-200";
    default:
      return "bg-gray-100 text-gray-800 border-gray-200";
  }
};

// --- LOADING & STATUS COMPONENTS ---

const useStatusCycling = () => {
  const [statusUpdates, setStatusUpdates] = useState<StatusUpdate[]>([]);
  const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null);

  const statusMessages = [
    "🔍 Parsing HTML for both products...",
    "🧠 Analyzing sentiment patterns...",
    "🏷️ Detecting key themes and issues...",
    "⚖️ Comparing product features...",
    "💡 Generating strategic insights...",
    "📊 Finalizing comparison report...",
  ];

  const start = useCallback(() => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      setStatusUpdates([
        { message: statusMessages[currentIndex], timestamp: Date.now() },
      ]);
      currentIndex = (currentIndex + 1) % statusMessages.length;
    }, 3000);
    setIntervalId(interval);
  }, []);

  const stop = useCallback(() => {
    if (intervalId) clearInterval(intervalId);
    setIntervalId(null);
    setStatusUpdates([
      { message: "✅ Analysis complete!", timestamp: Date.now() },
    ]);
  }, [intervalId]);

  const clear = useCallback(() => {
    if (intervalId) clearInterval(intervalId);
    setIntervalId(null);
    setStatusUpdates([]);
  }, [intervalId]);

  return { statusUpdates, start, stop, clear };
};

// Removed LoadingCarousel - using original implementation

// --- UI COMPONENTS ---

const ReconciledComparisonTable = ({
  productAThemes,
  productBThemes,
  productAName,
  productBName,
}: {
  productAThemes: Record<string, Theme>;
  productBThemes: Record<string, Theme>;
  productAName: string;
  productBName: string;
}) => {
  const themeNamesA = Object.keys(productAThemes);
  const themeNamesB = Object.keys(productBThemes);

  const { sharedThemes, uniqueToA, uniqueToB } = reconcileThemes(
    themeNamesA,
    themeNamesB
  );

  // Corrected Code (Safe)
  const getSentimentIcon = (sentiment: string) => {
    // Use "neutral" as a fallback if sentiment is missing
    switch ((sentiment || "neutral").toLowerCase()) {
      case "positive":
        return "✅";
      case "negative":
        return "❌";
      case "neutral":
        return "➖";
      default:
        return "❓";
    }
  };

  const getWinner = (sentimentA: string, sentimentB: string) => {
    const scoreMap = { positive: 3, neutral: 2, mixed: 1, negative: 0 };

    const safeSentimentA = (sentimentA || "neutral").toLowerCase();
    const safeSentimentB = (sentimentB || "neutral").toLowerCase();

    // The fallback "|| 2" now defaults to the neutral score.
    const scoreA = scoreMap[safeSentimentA as keyof typeof scoreMap] || 2;
    const scoreB = scoreMap[safeSentimentB as keyof typeof scoreMap] || 2;

    if (scoreA > scoreB) return productAName;
    if (scoreB > scoreA) return productBName;
    return "Tie";
  };

  if (sharedThemes.length === 0) {
    return (
      <div className="text-center py-8">
        <Trophy className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          No Directly Comparable Themes
        </h3>
        <p className="text-gray-500 mb-4">
          While both products have themes around similar topics, they focus on
          different aspects.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 mb-2">
              {productAName} Focus Areas:
            </h4>
            <div className="text-sm text-blue-700 space-y-1">
              {uniqueToA.slice(0, 4).map((theme, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full flex-shrink-0"></div>
                  <span>{formatThemeName(theme)}</span>
                </div>
              ))}
              {uniqueToA.length > 4 && (
                <div className="text-xs text-blue-600 italic">
                  +{uniqueToA.length - 4} more themes
                </div>
              )}
            </div>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="font-semibold text-green-800 mb-2">
              {productBName} Focus Areas:
            </h4>
            <div className="text-sm text-green-700 space-y-1">
              {uniqueToB.slice(0, 4).map((theme, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full flex-shrink-0"></div>
                  <span>{formatThemeName(theme)}</span>
                </div>
              ))}
              {uniqueToB.length > 4 && (
                <div className="text-xs text-green-600 italic">
                  +{uniqueToB.length - 4} more themes
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="overflow-x-auto border rounded-lg shadow-sm">
        <table className="min-w-full border-collapse bg-white">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-sm font-semibold text-gray-900 border-b border-gray-200 text-left">
                Theme Category
              </th>
              <th className="px-4 py-3 text-sm font-semibold text-gray-900 border-b border-gray-200 text-center">
                {productAName}
              </th>
              <th className="px-4 py-3 text-sm font-semibold text-gray-900 border-b border-gray-200 text-center">
                {productBName}
              </th>
              <th className="px-4 py-3 text-sm font-semibold text-gray-900 border-b border-gray-200 text-center">
                Winner
              </th>
            </tr>
          </thead>
          <tbody>
            {sharedThemes.map((shared, index) => {
              const themeA = productAThemes[shared.productA];
              const themeB = productBThemes[shared.productB];

              // --- FIX: ADD THIS SAFETY CHECK ---
              // This prevents a crash if a theme is missing from one of the products.
              if (!themeA || !themeB) {
                return null;
              }

              const winner = getWinner(
                themeA.overall_sentiment,
                themeB.overall_sentiment
              );

              return (
                <tr
                  key={index}
                  className="hover:bg-gray-50 border-b border-gray-200"
                >
                  <td className="px-4 py-3 text-sm font-medium text-gray-900 text-left">
                    {shared.canonical}
                  </td>
                  <td className="px-4 py-3 text-sm text-center text-gray-800">
                    <div className="flex items-center justify-center">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full ${getSentimentClass(
                          themeA.overall_sentiment,
                          "bg"
                        )} ${getSentimentClass(
                          themeA.overall_sentiment,
                          "text"
                        )}`}
                      >
                        {getSentimentIcon(themeA.overall_sentiment)}{" "}
                        {capitalize(themeA.overall_sentiment)}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-center text-gray-800">
                    <div className="flex items-center justify-center">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full ${getSentimentClass(
                          themeB.overall_sentiment,
                          "bg"
                        )} ${getSentimentClass(
                          themeB.overall_sentiment,
                          "text"
                        )}`}
                      >
                        {getSentimentIcon(themeB.overall_sentiment)}{" "}
                        {capitalize(themeB.overall_sentiment)}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-center">
                    <span
                      className={`font-semibold ${
                        winner === productAName
                          ? "text-blue-600"
                          : winner === productBName
                          ? "text-green-600"
                          : "text-gray-600"
                      }`}
                    >
                      {winner}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* --- FIX: ADDED INFORMATIVE SENTIMENT BADGES TO UNIQUE THEMES --- */}
      {(uniqueToA.length > 0 || uniqueToB.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6 border-t border-gray-200">
          {uniqueToA.length > 0 && (
            <div className="bg-blue-50/50 rounded-lg p-4 border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-3">
                {productAName} Unique Themes:
              </h4>
              <div className="space-y-2">
                {uniqueToA.map((themeName, i) => {
                  const themeDetails = productAThemes[themeName];
                  if (!themeDetails) return null;

                  return (
                    <div
                      key={i}
                      className="flex items-center justify-between gap-4"
                    >
                      <span className="text-sm text-blue-900">
                        {formatThemeName(themeName)}
                      </span>
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full whitespace-nowrap ${getSentimentClass(
                          themeDetails.overall_sentiment,
                          "bg"
                        )} ${getSentimentClass(
                          themeDetails.overall_sentiment,
                          "text"
                        )}`}
                      >
                        {getSentimentIcon(themeDetails.overall_sentiment)}{" "}
                        {capitalize(themeDetails.overall_sentiment)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          {uniqueToB.length > 0 && (
            <div className="bg-green-50/50 rounded-lg p-4 border border-green-200">
              <h4 className="font-semibold text-green-800 mb-3">
                {productBName} Unique Themes:
              </h4>
              <div className="space-y-2">
                {uniqueToB.map((themeName, i) => {
                  const themeDetails = productBThemes[themeName];
                  if (!themeDetails) return null;

                  return (
                    <div
                      key={i}
                      className="flex items-center justify-between gap-4"
                    >
                      <span className="text-sm text-green-900">
                        {formatThemeName(themeName)}
                      </span>
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full whitespace-nowrap ${getSentimentClass(
                          themeDetails.overall_sentiment,
                          "bg"
                        )} ${getSentimentClass(
                          themeDetails.overall_sentiment,
                          "text"
                        )}`}
                      >
                        {getSentimentIcon(themeDetails.overall_sentiment)}{" "}
                        {capitalize(themeDetails.overall_sentiment)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const StrategicInsightsCard = ({
  insights,
}: {
  insights?: StrategicInsights;
}) => {
  if (!insights) return null;

  return (
    <Card>
      <CardHeader className="border-b">
        <CardTitle className="text-2xl font-bold text-gray-800 flex items-center gap-3">
          <Lightbulb className="w-8 h-8 text-yellow-500" /> Strategic Insights
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        {/* CHANGED: Used ?. and ?? [] to safely map over potentially missing arrays */}
        {(insights?.competitive_advantages ?? []).length > 0 && (
          <div>
            <h3 className="font-semibold text-lg flex items-center gap-2 mb-3">
              <Trophy className="text-green-500" /> Competitive Advantages
            </h3>
            <ul className="space-y-2">
              {insights.competitive_advantages.map((adv, i) => (
                <li key={i} className="flex items-start gap-2 text-gray-700">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 shrink-0"></div>
                  <span>{adv}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {(insights?.areas_to_improve ?? []).length > 0 && (
          <div>
            <h3 className="font-semibold text-lg flex items-center gap-2 mb-3">
              <Shield className="text-red-500" /> Areas to Improve
            </h3>
            <ul className="space-y-2">
              {insights.areas_to_improve.map((area, i) => (
                <li key={i} className="flex items-start gap-2 text-gray-700">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2 shrink-0"></div>
                  <span>{area}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {(insights?.recommendations ?? []).length > 0 && (
          <div>
            <h3 className="font-semibold text-lg mb-3">
              Actionable Recommendations
            </h3>
            <div className="space-y-4">
              {insights.recommendations.map((rec, i) => (
                <div key={i} className="border rounded-lg p-4 bg-gray-50">
                  <div className="flex justify-between items-start gap-4 mb-3">
                    <p className="font-medium text-gray-800 flex-1">
                      {rec.recommendation}
                    </p>
                    <span
                      className={`text-xs font-bold px-3 py-1 rounded-full whitespace-nowrap border ${getPriorityClass(
                        rec.priority
                      )}`}
                    >
                      {capitalize(rec.priority)} Priority
                    </span>
                  </div>
                  <p className="text-sm text-gray-600">
                    <strong>Expected Impact:</strong> {rec.impact}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

const ProductAnalysisCard = ({
  analysis,
  productName,
}: {
  analysis?: ProductAnalysis;
  productName: string;
}) => {
  if (!analysis) return null; // Safe guard

  // CHANGED: Safely access nested properties with ?. and provide fallbacks with ??
  const positiveCount =
    analysis?.metrics?.sentiment_distribution?.["positive"] ?? 0;
    
  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {productName}
          <span className="text-sm font-normal text-gray-500">
            ({analysis?.metrics?.total_reviews ?? 0} reviews)
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 flex-grow">
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-center justify-center gap-1 text-2xl font-bold text-blue-600 mb-1">
              <Star className="w-5 h-5 text-yellow-400 fill-current" />
              {(analysis?.metrics?.average_rating ?? 0).toFixed(1)}
            </div>
            <div className="text-sm text-blue-800">Average Rating</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {positiveCount}
            </div>
            <div className="text-sm text-green-800">Positive Reviews</div>
          </div>
        </div>

        {analysis?.insights?.executive_summary && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="font-semibold text-blue-900 mb-2">
              Executive Summary
            </h4>
            <p className="text-sm text-blue-800">
              {analysis.insights.executive_summary}
            </p>
          </div>
        )}

        <Accordion type="multiple" className="w-full">
          <AccordionItem value="themes">
            <AccordionTrigger>
              Theme Analysis ({Object.keys(analysis?.themes ?? {}).length}{" "}
              themes)
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pt-4">
              {/* CHANGED: Safely map over themes with ?? {} */}
              {Object.entries(analysis?.themes ?? {}).map(
                ([theme, details]) => (
                  <div
                    key={theme}
                    className={`p-4 border-l-4 rounded-r-lg ${getSentimentClass(
                      details?.overall_sentiment,
                      "bg"
                    )} ${getSentimentClass(
                      details?.overall_sentiment,
                      "border"
                    )}`}
                  >
                    {/* ... inner theme rendering ... */}
                  </div>
                )
              )}
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="issues">
            <AccordionTrigger>
              Common Issues ({(analysis?.issues ?? []).length} identified)
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pt-4">
              {/* CHANGED: Safely map over issues with ?? [] */}
              {(analysis?.issues ?? []).length > 0 ? (
                analysis.issues.map((issue, index) => (
                  <div
                    key={`${issue.issue_name}-${index}`}
                    className="p-4 border-l-4 rounded-r-lg bg-red-50 border-red-300"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-red-900">
                        {formatThemeName(issue.issue_name)}
                      </h4>
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-xs font-bold px-2 py-1 rounded-full border ${getPriorityClass(
                            issue.severity
                          )}`}
                        >
                          {capitalize(issue.severity)}
                        </span>
                        <span className="text-xs text-gray-600">
                          {/* CHANGED: Used 'frequency' and provided a fallback */}
                          {issue.frequency ?? 1}{" "}
                          {(issue.frequency ?? 1) === 1
                            ? "mention"
                            : "mentions"}
                        </span>
                      </div>
                    </div>
                    <p className="text-sm text-gray-700 mb-3">
                      {issue.description}
                    </p>
                    {issue.example_quote && (
                      <blockquote className="text-sm italic text-red-700 pl-3 border-l-2 border-red-300">
                        "{issue.example_quote}"
                      </blockquote>
                    )}
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <AlertTriangle className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-500 italic">
                    No significant issues identified.
                  </p>
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
};

// --- MAIN COMPONENT ---

export default function CompetitorAnalyzer() {
  const [inputs, setInputs] = useState({
    html_a: "",
    product_name_a: "Your Product",
    html_b: "",
    product_name_b: "Competitor Product",
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<FullApiResponse | null>(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  useEffect(() => {
    document.title = "Competitor Analyzer";
  }, []);

  const {
    statusUpdates,
    start: startStatus,
    stop: stopStatus,
    clear: clearStatus,
  } = useStatusCycling();

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setInputs((prev) => ({ ...prev, [name]: value }));
  };

  const handleAnalyze = useCallback(async () => {
    if (!inputs.html_a.trim() || !inputs.html_b.trim()) {
      setError("Please provide HTML content for both products.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);
    startStatus();

    try {
      console.log("Sending request with data:", inputs); // Debug log

      const response = await fetch(`${API_BASE}api/compare`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(inputs),
      });

      console.log("Response status:", response.status); // Debug log
      console.log("Response headers:", response.headers); // Debug log

      if (!response.ok) {
        const errData = await response
          .json()
          .catch(() => ({ error: "Unknown error" }));
        throw new Error(
          errData.error || `HTTP error! status: ${response.status}`
        );
      }

      const data: FullApiResponse = await response.json();
      console.log("Received raw data:", data); // Debug log
      console.log("Product A themes:", data?.product_a?.themes); // Debug log
      console.log("Product B themes:", data?.product_b?.themes); // Debug log

      setResults(data);
    } catch (err) {
      console.error("Full error details:", err); // Debug log
      setError(
        `Analysis failed: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setLoading(false);
      stopStatus();
    }
  }, [inputs, startStatus, stopStatus]);

  const handleClear = useCallback(() => {
    setInputs({
      html_a: "",
      product_name_a: "Your Product",
      html_b: "",
      product_name_b: "Competitor Product",
    });
    setResults(null);
    setError("");
    clearStatus();
  }, [clearStatus]);

  return (
    <main className="p-4 md:p-8 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <header className="mb-6 flex items-center justify-between">
          <Button
            variant="outline"
            onClick={() => navigate("/")}
            className="flex items-center gap-2"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
            Back
          </Button>
          <h1 className="text-4xl font-bold">Competitor Review Analyzer</h1>
          <div className="w-[72px]"></div> {/* Spacer for centering */}
        </header>
        <header className="mb-8 text-center">
          <p className="text-lg text-gray-600">
            Compare two products side-by-side using HTML from review pages.
          </p>
        </header>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Product Inputs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="product_name_a" className="font-semibold">
                    Your Product (A)
                  </Label>
                  <Input
                    id="product_name_a"
                    name="product_name_a"
                    value={inputs.product_name_a}
                    onChange={handleInputChange}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="html_a">Product A HTML</Label>
                  <Textarea
                    id="html_a"
                    name="html_a"
                    value={inputs.html_a}
                    onChange={handleInputChange}
                    placeholder="Paste HTML for Your Product..."
                    className="h-48 font-mono text-xs mt-1"
                  />
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <Label htmlFor="product_name_b" className="font-semibold">
                    Competitor's Product (B)
                  </Label>
                  <Input
                    id="product_name_b"
                    name="product_name_b"
                    value={inputs.product_name_b}
                    onChange={handleInputChange}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="html_b">Product B HTML</Label>
                  <Textarea
                    id="html_b"
                    name="html_b"
                    value={inputs.html_b}
                    onChange={handleInputChange}
                    placeholder="Paste HTML for Competitor..."
                    className="h-48 font-mono text-xs mt-1"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-4 mt-6 pt-6 border-t">
              <Button onClick={handleClear} variant="outline">
                Clear All
              </Button>
              <Button onClick={handleAnalyze} disabled={loading} size="lg">
                {loading ? "Analyzing..." : "Analyze & Compare"}
              </Button>
            </div>

            {error && (
              <Alert variant="destructive" className="mt-4">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {loading && (
          <Card className="mb-8">
            <CardContent className="py-8">
              <div className="text-center">
                <FadeTransition
                  animationKey={statusUpdates[0]?.message || "Processing..."}
                >
                  <p className="text-lg font-medium text-gray-700 mb-4">
                    {statusUpdates[0]?.message || "Processing..."}
                  </p>
                </FadeTransition>
                <div className="bg-gray-200 rounded-full h-2 max-w-md mx-auto">
                  <div className="bg-blue-600 h-2 rounded-full animate-pulse w-1/3"></div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {results && (
          <div className="space-y-8">
            {/* Comparison Summary Table */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="text-blue-500" /> Head-to-Head Comparison
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ReconciledComparisonTable
                  productAThemes={results?.product_a?.themes ?? {}}
                  productBThemes={results?.product_b?.themes ?? {}}
                  productAName={inputs.product_name_a}
                  productBName={inputs.product_name_b}
                />
              </CardContent>
            </Card>

            {/* Strategic Insights */}
            <StrategicInsightsCard
              insights={results?.comparison?.strategic_insights}
            />

            {/* Individual Product Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <ProductAnalysisCard
                analysis={results.product_a}
                productName={inputs.product_name_a}
              />
              <ProductAnalysisCard
                analysis={results.product_b}
                productName={inputs.product_name_b}
              />
            </div>

            {/* Debug/Raw JSON */}
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="debug">
                <AccordionTrigger>
                  Debug: View Full JSON Response
                </AccordionTrigger>
                <AccordionContent>
                  {results ? (
                    <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-xs max-h-96 overflow-y-auto border">
                      {JSON.stringify(results, null, 2)}
                    </pre>
                  ) : (
                    <div className="p-4 text-gray-500 italic text-center">
                      No data to display
                    </div>
                  )}
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        )}
      </div>
    </main>
  );
}
