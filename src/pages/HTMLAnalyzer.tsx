import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { useState, useCallback } from "react";
import {
  ThumbsUp,
  ThumbsDown,
  Lightbulb,
  Minus,
  ClipboardCopy,
  Check,
} from "lucide-react";

// Types and Interfaces
type TokenUsage = { total_tokens: number; estimated_cost: number };
type AnalysisMetadata = {
  total_reviews: number;
  analysis_date: string;
  product_type: string;
  model_used: string;
  analysis_time_seconds: number;
  token_usage: TokenUsage;
};
type Recommendation = {
  recommendation: string;
  priority: "high" | "medium" | "low";
  impact: string;
  rationale: string;
};

type Issue = {
  issue_name: string;
  description: string;
  severity: "high" | "medium" | "low";
  frequency: number;
  example_quote: string;
};
type ThemeData = {
  sentiment: "positive" | "negative" | "mixed";
  confidence: number;
  positives: string[];
  negatives: string[];
  example_quote: string;
};
type Themes = { [themeName: string]: ThemeData };
type Metrics = {
  total_reviews: number;
  average_rating: number;
  sentiment_distribution: {
    positive?: number;
    negative?: number;
    neutral?: number;
  };
};
type Insights = {
  executive_summary: string;
  recommendations: Recommendation[];
  key_insights: string[];
  sentiment_breakdown?: {
    positive?: number;
    negative?: number;
    neutral?: number;
  };
};
type AnalysisResult = {
  analysis_metadata: AnalysisMetadata;
  insights: Insights;
  issues: Issue[];
  themes: Themes;
  metrics: Metrics;
  reviews: string[];
};

type SentimentType = "positive" | "negative" | "mixed";
type PriorityType = "high" | "medium" | "low";

// Utility Functions
const formatIssueName = (name: string = ""): string => {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const getSentimentColor = (sentiment: SentimentType): string => {
  const colorMap = {
    positive: "bg-green-100 text-green-800 border-green-300",
    negative: "bg-red-100 text-red-800 border-red-300",
    mixed: "bg-yellow-100 text-yellow-800 border-yellow-300",
  };
  return colorMap[sentiment] || "bg-gray-100 text-gray-800 border-gray-300";
};

const getPriorityColor = (priority: PriorityType): string => {
  switch (priority) {
    case "high":
      return "bg-red-100 text-red-800 border-red-300";
    case "medium":
      return "bg-yellow-100 text-yellow-800 border-yellow-300";
    case "low":
      return "bg-green-100 text-green-800 border-green-300";
    default:
      return "bg-gray-100 text-gray-800 border-gray-300";
  }
};

// Helper function to calculate sentiment from themes and issues
const calculateSentimentFromData = (results: AnalysisResult) => {
  // If sentiment_distribution exists in metrics, use it directly
  if (
    results.metrics?.sentiment_distribution &&
    (results.metrics.sentiment_distribution.positive ||
      results.metrics.sentiment_distribution.negative ||
      results.metrics.sentiment_distribution.neutral)
  ) {
    return results.metrics.sentiment_distribution;
  }

  const totalReviews =
    results.analysis_metadata?.total_reviews || results.reviews?.length || 0;

  if (totalReviews === 0) {
    return { positive: 0, negative: 0, neutral: 0 };
  }

  // Simple approach: count themes and issues directly
  let positive = 0;
  let negative = 0;
  let neutral = 0;

  if (results.themes) {
    Object.values(results.themes).forEach((theme) => {
      if (theme.sentiment === "positive") {
        positive += 5; // Each positive theme represents ~5 reviews
      } else if (theme.sentiment === "negative") {
        negative += 5; // Each negative theme represents ~5 reviews
      } else if (theme.sentiment === "mixed") {
        neutral += 3; // Mixed themes contribute to neutral
        negative += 2; // But also some negative
      }
    });
  }

  // Add issues as strong negative indicators
  if (results.issues) {
    negative += results.issues.length * 3; // Each issue represents ~3 negative reviews
  }

  // Scale to match total reviews
  const calculatedTotal = positive + negative + neutral;
  if (calculatedTotal > 0 && calculatedTotal !== totalReviews) {
    const scale = totalReviews / calculatedTotal;
    positive = Math.round(positive * scale);
    negative = Math.round(negative * scale);
    neutral = Math.round(neutral * scale);

    // Adjust for rounding errors
    const diff = totalReviews - (positive + negative + neutral);
    if (diff !== 0) {
      neutral += diff; // Add any difference to neutral
    }
  }

  return { positive, negative, neutral };
};

interface StatusUpdate {
  message: string;
  timestamp: number;
}

// Custom Hooks
const useStatusCycling = () => {
  const [statusUpdates, setStatusUpdates] = useState<StatusUpdate[]>([]);
  const [intervalId, setIntervalId] = useState<ReturnType<
    typeof setInterval
  > | null>(null);

  const statusMessages = [
    "🔍 Parsing HTML content...",
    "🎯 Extracting product information...",
    "📝 Identifying customer reviews...",
    "🧠 Analyzing sentiment patterns...",
    "🏷️ Detecting key themes and issues...",
    "💡 Looking for feature requests...",
    "✨ Extracting review highlights...",
    "📊 Generating insights...",
  ];

  const start = useCallback(() => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      setStatusUpdates([
        {
          message: statusMessages[currentIndex],
          timestamp: Date.now(),
        },
      ]);
      currentIndex = (currentIndex + 1) % statusMessages.length;
    }, 3000);

    setIntervalId(interval);
  }, [statusMessages]);

  const stop = useCallback(() => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
    setStatusUpdates([
      {
        message: "✅ Analysis complete!",
        timestamp: Date.now(),
      },
    ]);
  }, [intervalId]);

  const clear = useCallback(() => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
    setStatusUpdates([]);
  }, [intervalId]);

  return { statusUpdates, start, stop, clear };
};

// Simple FadeTransition component
const FadeTransition = ({
  children,
  animationKey,
}: {
  children: React.ReactNode;
  animationKey: number;
}) => {
  return (
    <div key={animationKey} className="transition-opacity duration-300">
      {children}
    </div>
  );
};

// LoadingCarousel component
const LoadingCarousel = ({ status }: { status: StatusUpdate | null }) => {
  return (
    <div className="flex items-center justify-center p-8 bg-blue-50 rounded-lg h-20">
      <FadeTransition animationKey={status?.timestamp || 0}>
        <span className="text-blue-800 text-center">
          {status?.message || "Processing..."}
        </span>
      </FadeTransition>
    </div>
  );
};

// Components
const ThemeDetailCard = ({
  themeName,
  themeData,
}: {
  themeName: string;
  themeData: ThemeData;
}) => {
  return (
    <Card key={themeName} className="flex-1 min-w-[300px]">
      <CardHeader>
        <div className="flex justify-between items-start">
          <CardTitle className="text-lg">
            {formatIssueName(themeName)}
          </CardTitle>
          <span
            className={`px-2 py-1 rounded-full text-xs font-medium border ${getSentimentColor(
              themeData.sentiment
            )}`}
          >
            {themeData.sentiment}
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {themeData.example_quote && (
          <div>
            <h5 className="text-xs font-semibold text-gray-500 uppercase mb-1">
              Example Quote
            </h5>
            <p className="border-l-4 border-blue-500 pl-3 text-sm text-gray-700 italic">
              "{themeData.example_quote}"
            </p>
          </div>
        )}
        {themeData.positives?.length > 0 && (
          <div>
            <h5 className="text-xs font-semibold text-green-600 uppercase mb-2">
              Positives
            </h5>
            <ul className="list-disc list-inside space-y-1">
              {themeData.positives.map((point, index) => (
                <li key={`pos-${index}`} className="text-sm text-gray-800">
                  {point}
                </li>
              ))}
            </ul>
          </div>
        )}
        {themeData.negatives?.length > 0 && (
          <div>
            <h5 className="text-xs font-semibold text-red-600 uppercase mb-2">
              Negatives
            </h5>
            <ul className="list-disc list-inside space-y-1">
              {themeData.negatives.map((point, index) => (
                <li key={`neg-${index}`} className="text-sm text-gray-800">
                  {point}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Fixed SentimentCard component
const SentimentCard = ({ results }: { results: AnalysisResult }) => {
  const sentiment = calculateSentimentFromData(results);

  if (
    !sentiment ||
    (sentiment.positive === 0 &&
      sentiment.negative === 0 &&
      sentiment.neutral === 0)
  ) {
    return null;
  }

  const items = [
    {
      type: "positive",
      count: sentiment.positive || 0,
      icon: ThumbsUp,
      bgColor: "bg-green-50",
      textColor: "text-green-800",
      iconColor: "text-green-600",
      countColor: "text-green-600",
    },
    {
      type: "neutral",
      count: sentiment.neutral || 0,
      icon: Minus,
      bgColor: "bg-gray-50",
      textColor: "text-gray-800",
      iconColor: "text-gray-600",
      countColor: "text-gray-600",
    },
    {
      type: "negative",
      count: sentiment.negative || 0,
      icon: ThumbsDown,
      bgColor: "bg-red-50",
      textColor: "text-red-800",
      iconColor: "text-red-600",
      countColor: "text-red-600",
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sentiment Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {items.map(
            ({
              type,
              count,
              icon: Icon,
              bgColor,
              textColor,
              iconColor,
              countColor,
            }) => (
              <div
                key={type}
                className={`flex items-center justify-between p-3 ${bgColor} rounded-md`}
              >
                <div className="flex items-center gap-2">
                  <Icon className={`w-4 h-4 ${iconColor}`} aria-hidden="true" />
                  <span
                    className={`text-sm font-medium ${textColor} capitalize`}
                  >
                    {type}
                  </span>
                </div>
                <span className={`text-2xl font-bold ${countColor}`}>
                  {count}
                </span>
              </div>
            )
          )}
        </div>
      </CardContent>
    </Card>
  );
};

const AnalysisOverviewCard = ({ results }: { results: AnalysisResult }) => {
  const metrics = [
    {
      label: "Total Reviews",
      value:
        results.analysis_metadata?.total_reviews ||
        results.reviews?.length ||
        0,
      bgColor: "bg-blue-50",
      textColor: "text-blue-800",
      countColor: "text-blue-600",
    },
    {
      label: "Themes Found",
      value: results.themes ? Object.keys(results.themes).length : 0,
      bgColor: "bg-green-50",
      textColor: "text-green-800",
      countColor: "text-green-600",
    },
    {
      label: "Issues Found",
      value: results.issues?.length || 0,
      bgColor: "bg-red-50",
      textColor: "text-red-800",
      countColor: "text-red-600",
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Analysis Overview</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {metrics.map(({ label, value, bgColor, textColor, countColor }) => (
            <div
              key={label}
              className={`flex items-center justify-between p-3 ${bgColor} rounded-md`}
            >
              <span className={`text-sm font-medium ${textColor}`}>
                {label}
              </span>
              <span className={`text-2xl font-bold ${countColor}`}>
                {value}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

const RecommendationCard = ({
  recommendation,
  index,
}: {
  recommendation: Recommendation;
  index: number;
}) => (
  <article className="border rounded-lg p-4" key={index}>
    <header className="flex items-start justify-between mb-3">
      <h3 className="font-semibold text-gray-800 flex-1">
        {recommendation.recommendation}
      </h3>
      <span
        className={`px-2 py-1 rounded-full text-xs font-medium border ${getPriorityColor(
          recommendation.priority
        )}`}
      >
        {recommendation.priority} priority
      </span>
    </header>

    <div className="space-y-2">
      <div>
        <p className="text-sm font-medium text-gray-700">Rationale:</p>
        <p className="text-sm text-gray-600">{recommendation.rationale}</p>
      </div>
    </div>
  </article>
);

const API_BASE = "http://localhost:5000/";

// Fixed generateReportText function
const generateReportText = (results: AnalysisResult): string => {
  const { insights, analysis_metadata, themes, issues } = results;

  const keyInsights =
    insights?.key_insights?.map((i) => `- ${i}`).join("\n") || "";

  const recommendations = (insights?.recommendations || [])
    .map(
      (r) =>
        `### ${r.recommendation} (Priority: ${r.priority})\nRationale: ${r.rationale}`
    )
    .join("\n\n");

  const themeDetails = Object.entries(themes || {})
    .map(
      ([themeName, t]) =>
        `### Theme: ${formatIssueName(themeName)} (Sentiment: ${
          t.sentiment
        })\n` +
        `Positives:\n${
          t.positives?.map((p) => `- ${p}`).join("\n") || "- None"
        }\n` +
        `Negatives:\n${
          t.negatives?.map((n) => `- ${n}`).join("\n") || "- None"
        }\n` +
        `Example Quote: "${t.example_quote || ""}"`
    )
    .join("\n\n");

  const issuesDetails = (issues || [])
    .map(
      (issue) =>
        `### Issue: ${formatIssueName(issue.issue_name)} (Severity: ${
          issue.severity
        })\n` +
        `Description: ${issue.description}\n` +
        `Example Quote: "${issue.example_quote || ""}"`
    )
    .join("\n\n");

  const sentiment = calculateSentimentFromData(results);

  return `# Analysis Report

## Executive Summary
${insights?.executive_summary || "No executive summary available."}

## Key Insights
${keyInsights || "No key insights found."}

## Analysis Metrics
- Total Reviews: ${analysis_metadata?.total_reviews || 0}
- Positive Sentiment: ${sentiment?.positive || 0}
- Negative Sentiment: ${sentiment?.negative || 0}
- Neutral Sentiment: ${sentiment?.neutral || 0}
- Themes Found: ${Object.keys(themes || {}).length}
- Issues Found: ${(issues || []).length}

## Actionable Recommendations
${recommendations || "No recommendations available."}

## Detailed Theme Analysis
${themeDetails || "No themes found."}

## Common Issues
${issuesDetails || "No issues found."}

---
Generated on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}`;
};

// Main Component
function HTMLAnalyzer() {
  const [htmlInput, setHtmlInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [copySuccess, setCopySuccess] = useState(false);
  const [error, setError] = useState("");

  const handleCopyReport = async () => {
    if (!results) return;
    try {
      const reportText = generateReportText(results);
      await navigator.clipboard.writeText(reportText);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error("Failed to copy report:", err);
      setError("Failed to copy report to clipboard");
    }
  };

  const {
    statusUpdates,
    start: startStatus,
    stop: stopStatus,
    clear: clearStatus,
  } = useStatusCycling();

  const handleAnalyze = useCallback(async () => {
    if (!htmlInput.trim()) {
      setError("Please enter HTML content to analyze");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);
    startStatus();

    try {
      const response = await fetch(`${API_BASE}api/analyze-html`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          html: htmlInput,
          product_type: "gaming controller",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      stopStatus();

      setTimeout(() => {
        setResults(data);
        setLoading(false);
      }, 1000);
    } catch (err) {
      stopStatus();
      setError(
        `Analysis failed: ${err instanceof Error ? err.message : String(err)}`
      );
      setLoading(false);
    }
  }, [htmlInput, startStatus, stopStatus]);

  const handleClear = useCallback(() => {
    setHtmlInput("");
    setResults(null);
    setError("");
    setCopySuccess(false);
    clearStatus();
  }, [clearStatus]);

  return (
    <main className="p-8 max-w-4xl mx-auto">
      <header className="mb-6">
        <h1 className="text-4xl font-bold">HTML Analyzer</h1>
      </header>

      <section aria-label="HTML Input">
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Manual HTML Analyzer</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label
                  htmlFor="html-input"
                  className="block text-sm font-medium mb-2"
                >
                  Paste HTML content here:
                </Label>
                <Textarea
                  id="html-input"
                  value={htmlInput}
                  onChange={(e) => setHtmlInput(e.target.value)}
                  placeholder="Paste your HTML content here..."
                  className="w-full h-48 resize-vertical font-mono text-sm"
                />
              </div>
              <div className="flex gap-3">
                <Button
                  onClick={handleAnalyze}
                  disabled={loading || !htmlInput.trim()}
                  className="flex-1"
                >
                  {loading ? "Analyzing..." : "Analyze HTML"}
                </Button>
                <Button onClick={handleClear} variant="outline">
                  Clear
                </Button>
              </div>
              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </div>
          </CardContent>
        </Card>
      </section>

      {loading && (
        <div className="my-6">
          <LoadingCarousel status={statusUpdates[0] || null} />
        </div>
      )}

      {results && (
        <section aria-label="Analysis Results" className="space-y-6">
          <div className="flex justify-end">
            <Button
              variant="outline"
              aria-label="Copy full analysis report to clipboard"
              onClick={handleCopyReport}
            >
              {copySuccess ? (
                <>
                  <Check className="mr-2 h-4 w-4" />
                  Copied!
                </>
              ) : (
                <>
                  <ClipboardCopy className="mr-2 h-4 w-4" />
                  Copy Report
                </>
              )}
            </Button>
          </div>

          {/* Executive Summary */}
          {results.insights?.executive_summary && (
            <Card>
              <CardHeader>
                <CardTitle>Executive Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 leading-relaxed">
                  {results.insights.executive_summary}
                </p>
              </CardContent>
            </Card>
          )}

          {/* Key Insights */}
          {results.insights?.key_insights?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Key Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {results.insights.key_insights.map((insight, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-3 p-3 bg-blue-50 rounded-md border-l-4 border-blue-400"
                    >
                      <Lightbulb className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                      <p className="text-blue-900 text-sm">{insight}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Analysis Metrics */}
          <section
            aria-label="Analysis Metrics"
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            <SentimentCard results={results} />
            <AnalysisOverviewCard results={results} />
            <Card>
              <CardHeader>
                <CardTitle>Analysis Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {results.analysis_metadata?.analysis_time_seconds && (
                    <div className="flex items-center justify-between p-3 bg-purple-50 rounded-md">
                      <span className="text-sm font-medium text-purple-800">
                        Analysis Time
                      </span>
                      <span className="text-lg font-bold text-purple-600">
                        {results.analysis_metadata.analysis_time_seconds.toFixed(
                          1
                        )}
                        s
                      </span>
                    </div>
                  )}
                  {results.analysis_metadata?.token_usage && (
                    <div className="flex items-center justify-between p-3 bg-orange-50 rounded-md">
                      <span className="text-sm font-medium text-orange-800">
                        Tokens Used
                      </span>
                      <span className="text-lg font-bold text-orange-600">
                        {results.analysis_metadata.token_usage.total_tokens?.toLocaleString()}
                      </span>
                    </div>
                  )}
                  {results.analysis_metadata?.model_used && (
                    <div className="p-3 bg-gray-50 rounded-md">
                      <span className="text-sm font-medium text-gray-800">
                        Model:{" "}
                      </span>
                      <span className="text-sm text-gray-600">
                        {results.analysis_metadata.model_used}
                      </span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </section>

          {/* Themes Section */}
          {results.themes && Object.keys(results.themes).length > 0 && (
            <section>
              <Card>
                <CardHeader>
                  <CardTitle>Detailed Theme Analysis</CardTitle>
                  <p className="text-sm text-gray-600">
                    Analysis based on{" "}
                    {results.analysis_metadata?.total_reviews || 0} reviews
                  </p>
                </CardHeader>
              </Card>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                {Object.entries(results.themes).map(
                  ([themeName, themeData]) => (
                    <ThemeDetailCard
                      key={themeName}
                      themeName={themeName}
                      themeData={themeData}
                    />
                  )
                )}
              </div>
            </section>
          )}

          {/* Recommendations */}
          {results.insights?.recommendations?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Actionable Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {results.insights.recommendations.map((rec, index) => (
                    <RecommendationCard
                      key={index}
                      recommendation={rec}
                      index={index}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Issues with Accordion */}
          {results.issues?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Common Issues</CardTitle>
              </CardHeader>
              <CardContent>
                <Accordion type="single" collapsible className="w-full">
                  {results.issues.map((issue, index) => (
                    <AccordionItem key={index} value={`item-${index}`}>
                      <AccordionTrigger className="p-4 hover:bg-red-50 rounded-md text-left">
                        <div className="flex justify-between items-center w-full">
                          <div className="flex-1">
                            <h4 className="font-semibold text-red-900">
                              {formatIssueName(issue.issue_name)}
                            </h4>
                            {issue.description && (
                              <p className="text-sm text-red-800 font-normal mt-1">
                                {issue.description}
                              </p>
                            )}
                          </div>
                          <span className="ml-4 text-xs px-2 py-1 bg-red-200 text-red-800 rounded-full whitespace-nowrap">
                            {issue.severity || "Unknown"}
                          </span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="p-4 border-t border-red-200">
                        <div className="space-y-2">
                          <h5 className="font-semibold text-sm text-gray-700 mb-2">
                            Example Quote:
                          </h5>
                          {issue.example_quote ? (
                            <blockquote className="p-3 bg-gray-100 rounded text-sm italic text-gray-700 border-l-4 border-gray-300">
                              "{issue.example_quote}"
                            </blockquote>
                          ) : (
                            <p className="text-sm text-gray-500 italic">
                              No quote found.
                            </p>
                          )}
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              </CardContent>
            </Card>
          )}

          {/* Full JSON Output */}
          <Card>
            <CardHeader>
              <CardTitle>Full JSON Output</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-100 p-4 rounded-md overflow-x-auto text-xs whitespace-pre-wrap">
                {JSON.stringify(results, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </section>
      )}
    </main>
  );
}

export default HTMLAnalyzer;
