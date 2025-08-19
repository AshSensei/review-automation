import React, { useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Lightbulb, Trophy, Shield, Star } from "lucide-react";
import { LoadingCarousel } from "../components/ui/LoadingCarousel"; // Assuming this component exists

// --- TYPESCRIPT INTERFACES ---
// Matches the JSON structure from the Python backend

interface Issue {
  issue_name: string;
  description: string;
  frequency: number;
  severity: string;
  example_quote: string;
  type: string;
}

interface Theme {
  mentions: number;
  sentiment: "positive" | "negative" | "neutral";
  example_quote: string;
  key_phrases?: string[];
}

interface Metrics {
  total_reviews: number;
  average_rating: number;
  rating_distribution: Record<string, number>;
  sentiment_distribution: Record<string, number>;
}

interface ProductAnalysis {
  product_type: string;
  themes: {
    themes: Record<string, Theme>;
    discovered_themes: string[];
    sample_size: number;
  };
  issues: Issue[];
  metrics: Metrics;
  analysis_metadata: {
    analysis_date: string;
    model_used: string;
    analysis_time_seconds: number;
  };
  reviews?: { review_text: string; review_rating: number | null }[];
}

interface StrategicRecommendation {
  recommendation: string;
  priority: "High" | "Medium" | "Low";
  impact: string;
}

interface StrategicInsights {
  competitive_advantages: string[];
  areas_to_improve: string[];
  recommendations: StrategicRecommendation[];
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

interface FullApiResponse {
  product_a: ProductAnalysis;
  product_b: ProductAnalysis;
  comparison: ComparisonData;
}

interface StatusUpdate {
  message: string;
  timestamp: number;
}

// --- UTILITY & HELPER FUNCTIONS ---

const formatThemeName = (name: string = ""): string => {
  return name
    .split("_")
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
    average: {
      text: "text-yellow-700",
      bg: "bg-yellow-50",
      border: "border-yellow-300",
    },
    good: {
      text: "text-blue-700",
      bg: "bg-blue-50",
      border: "border-blue-300",
    },
    excellent: {
      text: "text-purple-700",
      bg: "bg-purple-50",
      border: "border-purple-300",
    },
    poor: {
      text: "text-orange-700",
      bg: "bg-orange-50",
      border: "border-orange-300",
    },
  };
  return styles[sentimentLower]?.[element] || styles.neutral[element];
};

const getPriorityClass = (priority: string = "low") => {
  switch (priority.toLowerCase()) {
    case "high":
      return "bg-red-100 text-red-800";
    case "medium":
      return "bg-yellow-100 text-yellow-800";
    case "low":
      return "bg-green-100 text-green-800";
    default:
      return "bg-gray-100 text-gray-800";
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

// --- UI COMPONENTS ---

const KeyPhrasePopover = ({
  phrase,
  reviews,
}: {
  phrase: string;
  reviews: ProductAnalysis["reviews"];
}) => {
  const relatedReviews = reviews
    ? reviews.filter((review) =>
        review.review_text.toLowerCase().includes(phrase.toLowerCase())
      )
    : [];

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button className="px-2 py-1 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md text-xs transition-colors cursor-pointer border border-gray-300">
          {phrase}
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-80 max-h-96 overflow-y-auto">
        <div className="space-y-3">
          <h4 className="font-semibold text-sm">
            Reviews mentioning "{phrase}"
          </h4>
          {relatedReviews.length > 0 ? (
            relatedReviews.slice(0, 5).map((review, index) => (
              <blockquote
                key={index}
                className="border-l-4 border-blue-500 pl-3 py-2 bg-blue-50 text-sm text-gray-700 italic rounded-r-md"
              >
                "
                {review.review_text.length > 150
                  ? review.review_text.substring(0, 150) + "..."
                  : review.review_text}
                "
              </blockquote>
            ))
          ) : (
            <p className="text-sm text-gray-500 italic">No reviews found.</p>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
};

const RenderMarkdownTable = ({ markdown }: { markdown: string }) => {
  if (!markdown || typeof markdown !== "string")
    return <p>No comparison data available.</p>;

  const lines = markdown.trim().split("\n");
  if (lines.length < 2) return <p>No table data.</p>;

  // Parse header
  const headerCells = lines[0]
    .split("|")
    .map((cell) => cell.trim())
    .filter((cell) => cell !== "");

  // Parse data rows and normalize cell counts
  const rows = lines.slice(2).map((line) => {
    const cells = line
      .split("|")
      .map((cell) => cell.trim())
      .filter((cell) => cell !== "");
    // Pad or trim rows to match header length
    if (cells.length < headerCells.length) {
      return [...cells, ...Array(headerCells.length - cells.length).fill("")];
    } else if (cells.length > headerCells.length) {
      return cells.slice(0, headerCells.length);
    }
    return cells;
  });

  return (
    <div className="overflow-x-auto border rounded-md shadow-sm">
      <table className="min-w-full table-fixed border-collapse">
        <thead className="bg-gray-100">
          <tr>
            {headerCells.map((cell, index) => (
              <th
                key={index}
                className="px-4 py-2 text-sm font-semibold text-gray-800 border-b text-center"
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white">
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="hover:bg-gray-50">
              {row.map((cell, cellIndex) => (
                <td
                  key={cellIndex}
                  className="px-4 py-2 text-sm text-gray-700 border-t text-center break-words"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const StrategicInsightsCard = ({
  insights,
}: {
  insights: StrategicInsights;
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
        <div>
          <h3 className="font-semibold text-lg flex items-center gap-2 mb-2">
            <Trophy className="text-green-500" /> Competitive Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 pl-2">
            {insights.competitive_advantages?.map((adv, i) => (
              <li key={i}>{adv}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="font-semibold text-lg flex items-center gap-2 mb-2">
            <Shield className="text-red-500" /> Areas to Improve
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 pl-2">
            {insights.areas_to_improve?.map((area, i) => (
              <li key={i}>{area}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="font-semibold text-lg mb-3">
            Actionable Recommendations
          </h3>
          <div className="space-y-4">
            {insights.recommendations?.map((rec, i) => (
              <div key={i} className="border rounded-lg p-4 bg-gray-50/50">
                <div className="flex justify-between items-start gap-4">
                  <p className="font-medium text-gray-800 flex-1">
                    {rec.recommendation}
                  </p>
                  <span
                    className={`text-xs font-bold px-2 py-1 rounded-full whitespace-nowrap ${getPriorityClass(
                      rec.priority
                    )}`}
                  >
                    {rec.priority}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  <strong>Impact:</strong> {rec.impact}
                </p>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const ProductAnalysisCard = ({ analysis }: { analysis: ProductAnalysis }) => {
  if (!analysis) return null;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle>{analysis.product_type}</CardTitle>
        <p className="text-sm text-gray-500 pt-1">
          Based on {analysis.metrics.total_reviews} reviews
        </p>
      </CardHeader>
      <CardContent className="space-y-4 flex-grow">
        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4 text-center">
          <div className="p-3 bg-blue-50 rounded-lg border border-blue-100">
            <div className="text-3xl font-bold text-blue-600 flex items-center justify-center gap-1">
              <Star className="w-6 h-6 text-yellow-400 fill-current" />{" "}
              {analysis.metrics.average_rating.toFixed(2)}
            </div>
            <div className="text-sm text-blue-800">Avg. Rating</div>
          </div>
          <div className="p-3 bg-green-50 rounded-lg border border-green-100">
            <div className="text-3xl font-bold text-green-600">
              {analysis.metrics.sentiment_distribution["POSITIVE"] || 0}
            </div>
            <div className="text-sm text-green-800">Positive Reviews</div>
          </div>
        </div>

        {/* Deep Dive Accordions */}
        <Accordion type="multiple" className="w-full">
          <AccordionItem value="themes">
            <AccordionTrigger>
              Detailed Theme Analysis (
              {Object.keys(analysis.themes.themes).length})
            </AccordionTrigger>
            <AccordionContent className="space-y-3 pt-4">
              {Object.entries(analysis.themes.themes).map(
                ([theme, details]) => (
                  <div
                    key={theme}
                    className={`p-3 border-l-4 rounded-r-md ${getSentimentClass(
                      details.sentiment,
                      "bg"
                    )} ${getSentimentClass(details.sentiment, "border")}`}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-semibold">
                        {formatThemeName(theme)}
                      </h4>
                      <span
                        className={`text-xs font-bold px-2 py-1 rounded ${getSentimentClass(
                          details.sentiment,
                          "bg"
                        )} ${getSentimentClass(details.sentiment, "text")}`}
                      >
                        {capitalize(details.sentiment)}
                      </span>
                    </div>
                    <blockquote className="text-sm italic text-gray-600 mb-3">
                      "{details.example_quote}"
                    </blockquote>
                    {details.key_phrases && (
                      <div className="flex flex-wrap gap-2">
                        {details.key_phrases.map((phrase) => (
                          <KeyPhrasePopover
                            key={phrase}
                            phrase={phrase}
                            reviews={analysis.reviews || []}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )
              )}
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="issues">
            <AccordionTrigger>
              Common Issues ({analysis.issues.length})
            </AccordionTrigger>
            <AccordionContent className="space-y-3 pt-4">
              {analysis.issues.length > 0 ? (
                analysis.issues.map((issue) => (
                  <div
                    key={issue.issue_name}
                    className="p-3 border-l-4 rounded-r-md bg-red-50 border-red-300"
                  >
                    <div className="flex justify-between items-center mb-1">
                      <h4 className="font-semibold text-red-800">
                        {formatThemeName(issue.issue_name)}
                      </h4>
                      <span className="text-xs font-bold px-2 py-1 bg-red-200 text-red-800 rounded">
                        {issue.severity}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 mb-2">
                      {issue.description}
                    </p>
                    <blockquote className="text-sm italic text-gray-600">
                      "{issue.example_quote}"
                    </blockquote>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500 italic px-4">
                  No significant issues were identified from negative reviews.
                </p>
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

  useEffect(() => {
      // This line runs when the component mounts
      document.title = 'Competitor Analyzer';
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
      const response = await fetch("http://localhost:5001/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(
          errData.error || `HTTP error! status: ${response.status}`
        );
      }

      const data: FullApiResponse = await response.json();
      setResults(data);
    } catch (err) {
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
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-800">
            Competitor Review Analyzer
          </h1>
          <p className="text-lg text-gray-600 mt-2">
            Compare two products side-by-side using raw review HTML.
          </p>
        </header>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Product Inputs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <Label htmlFor="product_name_a" className="font-semibold">
                  Your Product (A)
                </Label>
                <Input
                  id="product_name_a"
                  name="product_name_a"
                  value={inputs.product_name_a}
                  onChange={handleInputChange}
                />
                <Label htmlFor="html_a">Product A HTML</Label>
                <Textarea
                  id="html_a"
                  name="html_a"
                  value={inputs.html_a}
                  onChange={handleInputChange}
                  placeholder="Paste HTML for Your Product..."
                  className="h-48 font-mono text-xs"
                />
              </div>
              <div className="space-y-4">
                <Label htmlFor="product_name_b" className="font-semibold">
                  Competitor's Product (B)
                </Label>
                <Input
                  id="product_name_b"
                  name="product_name_b"
                  value={inputs.product_name_b}
                  onChange={handleInputChange}
                />
                <Label htmlFor="html_b">Product B HTML</Label>
                <Textarea
                  id="html_b"
                  name="html_b"
                  value={inputs.html_b}
                  onChange={handleInputChange}
                  placeholder="Paste HTML for Competitor..."
                  className="h-48 font-mono text-xs"
                />
              </div>
            </div>
            <div className="flex items-center justify-end gap-4 mt-6 border-t pt-6">
              <Button onClick={handleClear} variant="ghost">
                Clear All
              </Button>
              <Button onClick={handleAnalyze} disabled={loading} size="lg">
                {loading ? "Analyzing..." : "Analyze & Compare"}
              </Button>
            </div>
            {error && (
              <Alert variant="destructive" className="mt-4">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {loading && <LoadingCarousel status={statusUpdates[0] || null} />}

        {results && (
          <div className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="text-blue-500" /> Comparison Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <RenderMarkdownTable
                  markdown={results.comparison.summary_table}
                />
              </CardContent>
            </Card>

            <StrategicInsightsCard
              insights={results.comparison.strategic_insights}
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <ProductAnalysisCard analysis={results.product_a} />
              <ProductAnalysisCard analysis={results.product_b} />
            </div>

            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="json">
                <AccordionTrigger>View Full JSON Response</AccordionTrigger>
                <AccordionContent>
                  <pre className="bg-gray-900 text-white p-4 rounded-md overflow-x-auto text-xs">
                    {JSON.stringify(results, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        )}
      </div>
    </main>
  );
}
