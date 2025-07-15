import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { useState } from "react";
import { ThumbsUp, ThumbsDown, Lightbulb, GitCompareArrows } from 'lucide-react';

// NOTE: The backend should be updated to provide 'related_quotes' for this to work
interface IssueObject {
  issue_name?: string;
  description?: string;
  example_quote?: string;
  related_quotes?: string[]; // Array to hold all quotes for the dropdown
  review_count?: number;
  severity?: string;
}

interface AnalysisResults {
  reviews?: string[];
  sentiment?: {
    positive: number;
    neutral: number;
    negative: number;
  };
  themes?: string[];
  issues?: (string | IssueObject)[];
  insights?: {
    executive_summary?: string;
    review_highlights?: {
      positive?: string[];
      negative?: string[];
    };
    feature_requests?: string[];
    comparative_analysis?: {
      competitor_product?: string;
      comparison_summary?: string;
      advantages?: string[];
      disadvantages?: string[];
    };
    [key: string]: any;
  };
  analysis_metadata?: {
    [key: string]: any;
  };
}


interface StatusUpdate {
  message: string;
  timestamp: number;
}

// Helper function to format snake_case to Title Case
const formatIssueName = (name: string = "") => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
};

function HTMLAnalyzer() {
  const [htmlInput, setHtmlInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState("");
  const [statusUpdates, setStatusUpdates] = useState<StatusUpdate[]>([]);
  const [statusCycleInterval, setStatusCycleInterval] = useState<NodeJS.Timeout | null>(null);
  const [selectedReview, setSelectedReview] = useState<string | null>(null);

  const startStatusCycling = () => {
    const updates = [
      "🔍 Parsing HTML content...",
      "🎯 Extracting product information...",
      "📝 Identifying customer reviews...",
      "🧠 Analyzing sentiment patterns...",
      "🏷️ Detecting key themes and issues...",
      "💡 Looking for feature requests...",
      "✨ Extracting review highlights...",
      "📊 Generating insights..."
    ];

    let currentIndex = 0;
    const interval = setInterval(() => {
        setStatusUpdates([{ message: updates[currentIndex], timestamp: Date.now() }]);
        currentIndex = (currentIndex + 1) % updates.length;
    }, 1200);
    
    setStatusCycleInterval(interval);
  };

  const stopStatusCycling = () => {
    if (statusCycleInterval) {
      clearInterval(statusCycleInterval);
      setStatusCycleInterval(null);
    }
    setStatusUpdates([{
      message: "✅ Analysis complete!",
      timestamp: Date.now()
    }]);
  };

  const findRelatedReview = (theme: string) => {
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
  };

  const handleAnalyze = async () => {
    if (!htmlInput.trim()) {
      setError("Please enter HTML content to analyze");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);
    setSelectedReview(null);
    startStatusCycling();

    try {
      const response = await fetch("http://localhost:5000/api/analyze-html", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ html: htmlInput, product_type: "gaming controller" }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      stopStatusCycling();
      
      setTimeout(() => {
        setResults(data);
        setLoading(false);
      }, 1000);
      
    } catch (err) {
      stopStatusCycling();
      setError(`Analysis failed: ${err instanceof Error ? err.message : String(err)}`);
      setLoading(false);
    }
  };

  const handleClear = () => {
    setHtmlInput("");
    setResults(null);
    setError("");
    setStatusUpdates([]);
    setSelectedReview(null);
    if (statusCycleInterval) {
      clearInterval(statusCycleInterval);
      setStatusCycleInterval(null);
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-4">HTML Analyzer</h1>

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

      {loading && statusUpdates.length > 0 && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Analysis in Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {statusUpdates.map((update, index) => (
                <div key={index} className="flex items-center gap-2 p-2 rounded-md bg-blue-50">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  <span className="text-sm text-blue-800">{update.message}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {results && (
        <div className="space-y-6">
          {results.insights?.executive_summary && (
            <Card>
              <CardHeader><CardTitle>Executive Summary</CardTitle></CardHeader>
              <CardContent><p className="text-gray-700 leading-relaxed">{results.insights.executive_summary}</p></CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Sentiment Analysis */}
            {results.sentiment && (
              <Card>
                <CardHeader>
                  <CardTitle>Sentiment Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 border rounded-md text-center bg-green-50">
                      <div className="text-3xl font-bold text-green-600">
                        {results.sentiment.positive || 0}
                      </div>
                      <div className="text-sm text-gray-600">Positive</div>
                    </div>
                    <div className="p-4 border rounded-md text-center bg-gray-50">
                      <div className="text-3xl font-bold text-gray-600">
                        {results.sentiment.neutral || 0}
                      </div>
                      <div className="text-sm text-gray-600">Neutral</div>
                    </div>
                    <div className="p-4 border rounded-md text-center bg-red-50">
                      <div className="text-3xl font-bold text-red-600">
                        {results.sentiment.negative || 0}
                      </div>
                      <div className="text-sm text-gray-600">Negative</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Analysis Overview */}
            <Card>
              <CardHeader>
                <CardTitle>Analysis Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-4 border rounded-md text-center bg-blue-50">
                    <div className="text-2xl font-bold text-blue-600">
                      {results.reviews?.length || 0}
                    </div>
                    <div className="text-sm text-gray-600">Total Reviews</div>
                  </div>
                  <div className="p-4 border rounded-md text-center bg-green-50">
                    <div className="text-2xl font-bold text-green-600">
                      {results.themes?.length || 0}
                    </div>
                    <div className="text-sm text-gray-600">Key Themes</div>
                  </div>
                  <div className="p-4 border rounded-md text-center bg-red-50">
                    <div className="text-2xl font-bold text-red-600">
                      {results.issues?.length || 0}
                    </div>
                    <div className="text-sm text-gray-600">Issues Found</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Key Themes */}
          {results.themes && results.themes.length > 0 && (
            <Card>
              <CardHeader><CardTitle>Key Themes (Click to see a related review)</CardTitle></CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {results.themes.map((theme, index) => (
                    <button
                      key={index}
                      onClick={() => findRelatedReview(theme)}
                      className="px-3 py-2 bg-blue-100 hover:bg-blue-200 rounded-md text-sm font-medium text-blue-800 transition-colors cursor-pointer"
                    >
                      {theme}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Issues with Accordion */}
          {results.issues && results.issues.length > 0 && (
            <Card>
              <CardHeader><CardTitle>Common Issues</CardTitle></CardHeader>
              <CardContent>
                <Accordion type="single" collapsible className="w-full">
                  {results.issues.map((issue, index) => {
                    if (typeof issue === 'string') {
                      return <div key={index} className="p-4 border-l-4 border-red-500 bg-red-50 rounded-r-md">{issue}</div>;
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
                                    {issueObj.description && <p className="text-sm text-red-800 font-normal mt-1">{issueObj.description}</p>}
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
                                <div key={qIndex} className="p-3 bg-gray-100 rounded text-sm italic text-gray-700 border-l-4 border-gray-300">
                                  "{quote}"
                                </div>
                              ))
                            ) : (
                                <p className="text-sm text-gray-500 italic">No specific quotes were found for this issue.</p>
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

          {/* Additional Insights */}
          {results.insights && (
            <Card>
              <CardHeader><CardTitle>Additional Insights</CardTitle></CardHeader>
              <CardContent className="space-y-6">
                {results.insights.review_highlights && (
                   <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                          <h3 className="flex items-center text-lg font-semibold text-green-700 mb-2"><ThumbsUp className="w-5 h-5 mr-2" />Positive Highlights</h3>
                          <div className="space-y-3">
                              {results.insights.review_highlights.positive?.map((quote, index) => (
                                  <div key={index} className="p-3 border-l-4 border-green-500 bg-green-50 rounded-r-md text-sm text-green-900 italic">"{quote}"</div>
                              ))}
                          </div>
                      </div>
                      <div>
                          <h3 className="flex items-center text-lg font-semibold text-red-700 mb-2"><ThumbsDown className="w-5 h-5 mr-2" />Negative Highlights</h3>
                          <div className="space-y-3">
                              {results.insights.review_highlights.negative?.map((quote, index) => (
                                  <div key={index} className="p-3 border-l-4 border-red-500 bg-red-50 rounded-r-md text-sm text-red-900 italic">"{quote}"</div>
                              ))}
                          </div>
                      </div>
                    </div>
                )}
                {results.insights.feature_requests && results.insights.feature_requests.length > 0 && (
                    <div>
                        <h3 className="flex items-center text-lg font-semibold text-yellow-800 mb-2"><Lightbulb className="w-5 h-5 mr-2" />Feature Requests</h3>
                        <div className="space-y-3">
                            {results.insights.feature_requests.map((request, index) => (
                                <div key={index} className="p-3 bg-yellow-50 border-l-4 border-yellow-400 rounded-r-md text-sm text-yellow-900">"{request}"</div>
                            ))}
                        </div>
                    </div>
                )}
                {results.insights.comparative_analysis && (
                  <div>
                    <h3 className="flex items-center text-lg font-semibold text-purple-800 mb-2"><GitCompareArrows className="w-5 h-5 mr-2" />Comparative Analysis</h3>
                    <div className="p-4 bg-purple-50 border-l-4 border-purple-400 rounded-r-md space-y-3">
                      <p className="text-sm text-purple-900">{results.insights.comparative_analysis.comparison_summary}</p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                              <h4 className="font-semibold text-green-700 mb-1">Advantages</h4>
                              <ul className="list-disc list-inside space-y-1 text-sm text-green-800">
                                  {results.insights.comparative_analysis.advantages?.map((adv, index) => <li key={index}>{adv}</li>)}
                              </ul>
                          </div>
                          <div>
                              <h4 className="font-semibold text-red-700 mb-1">Disadvantages</h4>
                              <ul className="list-disc list-inside space-y-1 text-sm text-red-800">
                                  {results.insights.comparative_analysis.disadvantages?.map((dis, index) => <li key={index}>{dis}</li>)}
                              </ul>
                          </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Analysis Metadata */}
          {results.analysis_metadata && (
            <Card>
              <CardHeader><CardTitle>Analysis Details</CardTitle></CardHeader>
              <CardContent>
                <pre className="bg-gray-100 p-3 rounded-md overflow-x-auto text-xs">
                  {JSON.stringify(results.analysis_metadata, null, 2)}
                </pre>
              </CardContent>
            </Card>
          )}

          {selectedReview && (
            <Card id="selected-review">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Related Review
                  <Button variant="ghost" size="sm" onClick={() => setSelectedReview(null)}>Close</Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="p-4 border-l-4 border-blue-500 bg-blue-50 rounded-r-md">
                  <p className="text-gray-800 italic">"{selectedReview}"</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

export default HTMLAnalyzer;