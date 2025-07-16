import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { useState, useCallback, useEffect } from "react";
import { ThumbsUp, ThumbsDown, Lightbulb, X } from 'lucide-react';

// Types and Interfaces
interface IssueObject {
  issue_name?: string;
  description?: string;
  example_quote?: string;
  related_quotes?: string[];
  review_count?: number;
  severity?: string;
}

interface ThemeDetails {
  confidence: number;
  example_quote: string;
  key_phrases: string[];
  mentions: number;
  sentiment: "positive" | "negative" | "mixed";
}

interface Recommendation {
  recommendation: string;
  priority: "high" | "medium" | "low";
  rationale: string;
  impact: string;
}

interface AnalysisResults {
  reviews?: string[];
  sentiment?: {
    positive: number;
    neutral: number;
    negative: number;
  };
  themes?: {
    discovered_themes: string[];
    sample_size: number;
    themes: { [key: string]: ThemeDetails };
  };
  issues?: (string | IssueObject)[];
  insights?: {
    executive_summary?: string;
    key_insights?: string[];
    recommendations?: Recommendation[];
  };
  analysis_metadata?: {
    analysis_date?: string;
    analysis_time_seconds?: number;
    model_used?: string;
    product_type?: string;
    token_usage?: {
      estimated_cost?: number;
      total_tokens?: number;
    };
    total_reviews?: number;
  };
  summary?: string;
}

interface StatusUpdate {
  message: string;
  timestamp: number;
}

type SentimentType = "positive" | "negative" | "mixed";
type PriorityType = "high" | "medium" | "low";

// Custom Hooks
const useStatusCycling = () => {
  const [statusUpdates, setStatusUpdates] = useState<StatusUpdate[]>([]);
  const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null);

  const statusMessages = [
    "🔍 Parsing HTML content...",
    "🎯 Extracting product information...",
    "📝 Identifying customer reviews...",
    "🧠 Analyzing sentiment patterns...",
    "🏷️ Detecting key themes and issues...",
    "💡 Looking for feature requests...",
    "✨ Extracting review highlights...",
    "📊 Generating insights..."
  ];

  const start = useCallback(() => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      setStatusUpdates([{ 
        message: statusMessages[currentIndex], 
        timestamp: Date.now() 
      }]);
      currentIndex = (currentIndex + 1) % statusMessages.length;
    }, 1200);
    
    setIntervalId(interval);
  }, []);

  const stop = useCallback(() => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
    setStatusUpdates([{
      message: "✅ Analysis complete!",
      timestamp: Date.now()
    }]);
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

// Utility Functions
const formatIssueName = (name: string = ""): string => {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const getSentimentColor = (sentiment: SentimentType): string => {
  const colors = {
    positive: 'bg-green-100 text-green-800 border-green-300',
    negative: 'bg-red-100 text-red-800 border-red-300',
    mixed: 'bg-yellow-100 text-yellow-800 border-yellow-300'
  };
  return colors[sentiment] || 'bg-gray-100 text-gray-800 border-gray-300';
};

const getPriorityColor = (priority: PriorityType): string => {
  const colors = {
    high: 'bg-red-100 text-red-800 border-red-300',
    medium: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    low: 'bg-green-100 text-green-800 border-green-300'
  };
  return colors[priority] || 'bg-gray-100 text-gray-800 border-gray-300';
};

// Components
const StatusIndicator = ({ updates }: { updates: StatusUpdate[] }) => (
  <Card className="mb-6">
    <CardHeader>
      <CardTitle>Analysis in Progress</CardTitle>
    </CardHeader>
    <CardContent>
      <div className="space-y-2">
        {updates.map((update, index) => (
          <div key={`${update.timestamp}-${index}`} className="flex items-center gap-2 p-2 rounded-md bg-blue-50">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" aria-hidden="true"></div>
            <span className="text-sm text-blue-800">{update.message}</span>
          </div>
        ))}
      </div>
    </CardContent>
  </Card>
);

const SentimentCard = ({ sentiment }: { sentiment: AnalysisResults['sentiment'] }) => {
  if (!sentiment) return null;

  const items = [
    { type: 'positive', count: sentiment.positive, icon: ThumbsUp, color: 'green' },
    { type: 'neutral', count: sentiment.neutral, icon: null, color: 'gray' },
    { type: 'negative', count: sentiment.negative, icon: ThumbsDown, color: 'red' }
  ] as const;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sentiment Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {items.map(({ type, count, icon: Icon, color }) => (
            <div key={type} className={`flex items-center justify-between p-3 bg-${color}-50 rounded-md`}>
              <div className="flex items-center gap-2">
                {Icon ? (
                  <Icon className={`w-4 h-4 text-${color}-600`} aria-hidden="true" />
                ) : (
                  <span className={`w-4 h-4 bg-${color}-400 rounded-full`} aria-hidden="true"></span>
                )}
                <span className={`text-sm font-medium text-${color}-800 capitalize`}>{type}</span>
              </div>
              <span className={`text-2xl font-bold text-${color}-600`}>{count}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

const AnalysisOverviewCard = ({ results }: { results: AnalysisResults }) => {
  const metrics = [
    {
      label: 'Total Reviews',
      value: results.analysis_metadata?.total_reviews || results.reviews?.length || 0,
      color: 'blue'
    },
    {
      label: 'Themes Found',
      value: results.themes?.discovered_themes?.length || 0,
      color: 'green'
    },
    {
      label: 'Issues Found',
      value: results.issues?.length || 0,
      color: 'red'
    }
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Analysis Overview</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {metrics.map(({ label, value, color }) => (
            <div key={label} className={`flex items-center justify-between p-3 bg-${color}-50 rounded-md`}>
              <span className={`text-sm font-medium text-${color}-800`}>{label}</span>
              <span className={`text-2xl font-bold text-${color}-600`}>{value}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

const ThemeButton = ({ 
  theme, 
  onClick, 
  className = "" 
}: { 
  theme: string; 
  onClick: () => void; 
  className?: string;
}) => (
  <button
    onClick={onClick}
    className={`px-3 py-2 bg-blue-100 hover:bg-blue-200 rounded-md text-sm font-medium text-blue-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${className}`}
  >
    {formatIssueName(theme)}
  </button>
);

const RecommendationCard = ({ recommendation, index }: { recommendation: Recommendation; index: number }) => (
  <article className="border rounded-lg p-4">
    <header className="flex items-start justify-between mb-3">
      <h3 className="font-semibold text-gray-800 flex-1">{recommendation.recommendation}</h3>
      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getPriorityColor(recommendation.priority)}`}>
        {recommendation.priority} priority
      </span>
    </header>
    
    <div className="space-y-2">
      <div>
        <p className="text-sm font-medium text-gray-700">Rationale:</p>
        <p className="text-sm text-gray-600">{recommendation.rationale}</p>
      </div>
      <div>
        <p className="text-sm font-medium text-gray-700">Expected Impact:</p>
        <p className="text-sm text-gray-600">{recommendation.impact}</p>
      </div>
    </div>
  </article>
);

const SelectedReviewCard = ({ 
  review, 
  onClose 
}: { 
  review: string; 
  onClose: () => void;
}) => (
  <Card id="selected-review">
    <CardHeader>
      <div className="flex items-center justify-between">
        <CardTitle>Related Review</CardTitle>
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={onClose}
          aria-label="Close review"
        >
          <X className="w-4 h-4" />
        </Button>
      </div>
    </CardHeader>
    <CardContent>
      <blockquote className="p-4 border-l-4 border-blue-500 bg-blue-50 rounded-r-md">
        <p className="text-gray-800 italic">"{review}"</p>
      </blockquote>
    </CardContent>
  </Card>
);

// Main Component
function HTMLAnalyzer() {
  const [htmlInput, setHtmlInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState("");
  const [selectedReview, setSelectedReview] = useState<string | null>(null);
  
  const { statusUpdates, start: startStatus, stop: stopStatus, clear: clearStatus } = useStatusCycling();

  const findRelatedReview = useCallback((theme: string) => {
    const matchingReview = results?.reviews?.find(review => 
      review.toLowerCase().includes(theme.toLowerCase())
    );
    
    if (matchingReview) {
      setSelectedReview(matchingReview);
      setTimeout(() => {
        document.getElementById('selected-review')?.scrollIntoView({ 
          behavior: 'smooth' 
        });
      }, 100);
    }
  }, [results?.reviews]);

  const handleAnalyze = useCallback(async () => {
    if (!htmlInput.trim()) {
      setError("Please enter HTML content to analyze");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);
    setSelectedReview(null);
    startStatus();

    try {
      const response = await fetch("http://localhost:5000/api/analyze-html", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ html: htmlInput, product_type: "gaming controller" }),
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
      setError(`Analysis failed: ${err instanceof Error ? err.message : String(err)}`);
      setLoading(false);
    }
  }, [htmlInput, startStatus, stopStatus]);

  const handleClear = useCallback(() => {
    setHtmlInput("");
    setResults(null);
    setError("");
    setSelectedReview(null);
    clearStatus();
  }, [clearStatus]);

  const handleCloseReview = useCallback(() => {
    setSelectedReview(null);
  }, []);

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
                <Label htmlFor="html-input" className="block text-sm font-medium mb-2">
                  Paste HTML content here:
                </Label>
                <Textarea
                  id="html-input"
                  value={htmlInput}
                  onChange={(e) => setHtmlInput(e.target.value)}
                  placeholder="Paste your HTML content here..."
                  className="w-full h-48 resize-vertical font-mono text-sm"
                  aria-describedby={error ? "html-input-error" : undefined}
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
                <Alert variant="destructive" role="alert" id="html-input-error">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </div>
          </CardContent>
        </Card>
      </section>

      {loading && statusUpdates.length > 0 && (
        <StatusIndicator updates={statusUpdates} />
      )}

      {results && (
        <section aria-label="Analysis Results" className="space-y-6">
          {/* Executive Summary */}
          {(results.insights?.executive_summary || results.summary) && (
            <Card>
              <CardHeader>
                <CardTitle>Executive Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 leading-relaxed">
                  {results.insights?.executive_summary || results.summary}
                </p>
              </CardContent>
            </Card>
          )}

          {/* Key Insights */}
          {results.insights?.key_insights && results.insights.key_insights.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Key Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {results.insights.key_insights.map((insight, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-blue-50 rounded-md border-l-4 border-blue-400">
                      <Lightbulb className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" aria-hidden="true" />
                      <p className="text-blue-900 text-sm">{insight}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Analysis Metrics */}
          <section aria-label="Analysis Metrics" className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <SentimentCard sentiment={results.sentiment} />
            <AnalysisOverviewCard results={results} />

            {/* Analysis Metadata */}
            {results.analysis_metadata && (
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {results.analysis_metadata.analysis_time_seconds && (
                      <div className="flex items-center justify-between p-3 bg-purple-50 rounded-md">
                        <span className="text-sm font-medium text-purple-800">Analysis Time</span>
                        <span className="text-lg font-bold text-purple-600">
                          {results.analysis_metadata.analysis_time_seconds.toFixed(1)}s
                        </span>
                      </div>
                    )}
                    {results.analysis_metadata.token_usage && (
                      <div className="flex items-center justify-between p-3 bg-orange-50 rounded-md">
                        <span className="text-sm font-medium text-orange-800">Tokens Used</span>
                        <span className="text-lg font-bold text-orange-600">
                          {results.analysis_metadata.token_usage.total_tokens?.toLocaleString()}
                        </span>
                      </div>
                    )}
                    {results.analysis_metadata.model_used && (
                      <div className="p-3 bg-gray-50 rounded-md">
                        <span className="text-sm font-medium text-gray-800">Model: </span>
                        <span className="text-sm text-gray-600">{results.analysis_metadata.model_used}</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </section>

          {/* Detailed Theme Analysis */}
          {results.themes?.themes && Object.keys(results.themes.themes).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Detailed Theme Analysis</CardTitle>
                <p className="text-sm text-gray-600">
                  Analysis based on {results.themes.sample_size} reviews
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(results.themes.themes).map(([themeName, themeData]) => (
                    <article key={themeName} className="border rounded-lg p-4">
                      <header className="flex items-center justify-between mb-3">
                        <h3 className="font-semibold text-lg text-gray-800">
                          {formatIssueName(themeName)}
                        </h3>
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getSentimentColor(themeData.sentiment)}`}>
                            {themeData.sentiment}
                          </span>
                          <span className="text-sm text-gray-600">
                            {themeData.mentions} mentions
                          </span>
                        </div>
                      </header>
                      
                      <div className="mb-3">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-medium text-gray-700">Confidence:</span>
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-500 h-2 rounded-full" 
                              style={{ width: `${themeData.confidence * 100}%` }}
                              role="progressbar"
                              aria-valuenow={themeData.confidence * 100}
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-label={`Confidence: ${(themeData.confidence * 100).toFixed(0)}%`}
                            />
                          </div>
                          <span className="text-sm text-gray-600">{(themeData.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>

                      <div className="mb-3">
                        <p className="text-sm text-gray-700 font-medium mb-2">Example Quote:</p>
                        <blockquote className="italic text-gray-600 border-l-4 border-gray-300 pl-3 text-sm">
                          "{themeData.example_quote}"
                        </blockquote>
                      </div>

                      <div>
                        <p className="text-sm text-gray-700 font-medium mb-2">Key Phrases:</p>
                        <div className="flex flex-wrap gap-2">
                          {themeData.key_phrases.map((phrase, index) => (
                            <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 rounded-md text-xs">
                              {phrase}
                            </span>
                          ))}
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Quick Theme Overview */}
          {results.themes?.discovered_themes && results.themes.discovered_themes.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Quick Theme Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {results.themes.discovered_themes.map((theme, index) => (
                    <ThemeButton
                      key={index}
                      theme={theme}
                      onClick={() => findRelatedReview(theme)}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Recommendations */}
          {results.insights?.recommendations && results.insights.recommendations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Actionable Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {results.insights.recommendations.map((rec, index) => (
                    <RecommendationCard key={index} recommendation={rec} index={index} />
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Issues with Accordion */}
          {results.issues && results.issues.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Common Issues</CardTitle>
              </CardHeader>
              <CardContent>
                <Accordion type="single" collapsible className="w-full">
                  {results.issues.map((issue, index) => {
                    if (typeof issue === 'string') {
                      return (
                        <div key={index} className="p-4 border-l-4 border-red-500 bg-red-50 rounded-r-md">
                          {issue}
                        </div>
                      );
                    }
                    
                    const issueObj = issue as IssueObject;
                    const quotes = issueObj.related_quotes || (issueObj.example_quote ? [issueObj.example_quote] : []);
                    
                    return (
                      <AccordionItem key={index} value={`item-${index}`}>
                        <AccordionTrigger className="p-4 hover:bg-red-50 rounded-md text-left">
                          <div className="flex justify-between items-center w-full">
                            <div className="flex-1">
                              <h4 className="font-semibold text-red-900">
                                {formatIssueName(issueObj.issue_name)}
                              </h4>
                              {issueObj.description && (
                                <p className="text-sm text-red-800 font-normal mt-1">
                                  {issueObj.description}
                                </p>
                              )}
                            </div>
                            <span className="ml-4 text-xs px-2 py-1 bg-red-200 text-red-800 rounded-full whitespace-nowrap">
                              {issueObj.severity || 'Unknown'}
                            </span>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="p-4 border-t border-red-200">
                          <div className="space-y-2">
                            <h5 className="font-semibold text-sm text-gray-700 mb-2">
                              {`Mentioned in ${issueObj.review_count || quotes.length} review${(issueObj.review_count ?? quotes.length) !== 1 ? 's' : ''}:`}
                            </h5>
                            {quotes.length > 0 ? (
                              quotes.map((quote, qIndex) => (
                                <blockquote key={qIndex} className="p-3 bg-gray-100 rounded text-sm italic text-gray-700 border-l-4 border-gray-300">
                                  "{quote}"
                                </blockquote>
                              ))
                            ) : (
                              <p className="text-sm text-gray-500 italic">
                                No specific quotes were found for this issue.
                              </p>
                            )}
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    );
                  })}
                </Accordion>
              </CardContent>
            </Card>
          )}

          {selectedReview && (
            <SelectedReviewCard review={selectedReview} onClose={handleCloseReview} />
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