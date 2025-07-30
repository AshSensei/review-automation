import React, { useState, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  Brain, 
  Code, 
  TrendingUp, 
  AlertTriangle, 
  Lightbulb, 
  Target,
  Clock,
  DollarSign,
  BarChart3,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  Activity,
  Download
} from 'lucide-react';

// Types based on your Python backend
interface AnalysisRequest {
  html: string;
  product_type: string;
  target_part?: string;
  use_llm_keywords?: boolean;
}

interface ThemeDetails {
  confidence: number;
  example_quote: string;
  key_phrases: string[];
  mentions: number;
  sentiment: "positive" | "negative" | "mixed";
}

interface Issue {
  issue_name: string;
  description: string;
  frequency: number;
  severity: "high" | "medium" | "low";
  example_quote: string;
  type?: string;
  target_part?: string;
}

interface Recommendation {
  recommendation: string;
  priority: "high" | "medium" | "low";
  rationale: string;
  impact: string;
}

interface AnalysisResults {
  themes?: {
    themes: { [key: string]: ThemeDetails };
    discovered_themes: string[];
    sample_size: number;
    target_part?: string;
    filtered_review_count?: number;
    original_review_count?: number;
  };
  issues?: Issue[];
  metrics?: {
    total_reviews: number;
    average_rating: number;
    rating_distribution: { [key: string]: number };
    average_review_length: number;
    sentiment_distribution: { [key: string]: number };
    average_sentiment_confidence: number;
    analysis_quality: string;
  };
  insights?: {
    executive_summary: string;
    recommendations: Recommendation[];
    key_insights: string[];
  };
  analysis_metadata?: {
    total_reviews: number;
    analysis_date: string;
    product_type: string;
    target_part?: string;
    model_used: string;
    analysis_time_seconds: number;
    token_usage: {
      total_tokens: number;
      estimated_cost_usd: number;
    };
  };
}

const API_BASE_URL = 'http://localhost:5000/api';

// Common product features for dropdown
const COMMON_FEATURES = [
  { value: "battery", label: "Battery Life" },
  { value: "build_quality", label: "Build Quality" },
  { value: "comfort", label: "Comfort & Ergonomics" },
  { value: "connectivity", label: "Connectivity" },
  { value: "design", label: "Design & Aesthetics" },
  { value: "durability", label: "Durability" },
  { value: "ease_of_use", label: "Ease of Use" },
  { value: "microphone", label: "Microphone Quality" },
  { value: "performance", label: "Performance" },
  { value: "price", label: "Price & Value" },
  { value: "responsiveness", label: "Responsiveness" },
  { value: "software", label: "Software & Drivers" },
  { value: "sound_quality", label: "Sound Quality" },
  { value: "wireless", label: "Wireless Features" },
  { value: "other", label: "Other (specify)" }
];

// Utility functions
const formatThemeName = (name: string) => {
  return name.split('_').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
};

const getSeverityColor = (severity: "high" | "medium" | "low") => {
  switch (severity) {
    case 'high': return 'bg-red-100 text-red-800 border-red-300';
    case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    case 'low': return 'bg-green-100 text-green-800 border-green-300';
    default: return 'bg-gray-100 text-gray-800 border-gray-300';
  }
};

const getPriorityColor = (priority: "high" | "medium" | "low") => {
  switch (priority) {
    case 'high': return 'bg-red-100 text-red-800 border-red-300';
    case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    case 'low': return 'bg-green-100 text-green-800 border-green-300';
    default: return 'bg-gray-100 text-gray-800 border-gray-300';
  }
};

const getSentimentColor = (sentiment: "positive" | "negative" | "mixed") => {
  switch (sentiment) {
    case 'positive': return 'bg-green-100 text-green-800 border-green-300';
    case 'negative': return 'bg-red-100 text-red-800 border-red-300';
    case 'mixed': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    default: return 'bg-gray-100 text-gray-800 border-gray-300';
  }
};

// Components
const AnalysisOverviewCard = ({ results }: { results: AnalysisResults }) => {
  const metrics = [
    {
      label: 'Total Reviews',
      value: results.metrics?.total_reviews || 0,
      icon: MessageSquare,
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      textColor: 'text-blue-700',
      iconColor: 'text-blue-600',
      valueColor: 'text-blue-900'
    },
    {
      label: 'Avg Rating',
      value: results.metrics?.average_rating?.toFixed(1) || 'N/A',
      icon: BarChart3,
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      textColor: 'text-green-700',
      iconColor: 'text-green-600',
      valueColor: 'text-green-900'
    },
    {
      label: 'Themes Found',
      value: results.themes?.discovered_themes?.length || 0,
      icon: TrendingUp,
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200',
      textColor: 'text-purple-700',
      iconColor: 'text-purple-600',
      valueColor: 'text-purple-900'
    },
    {
      label: 'Issues Found',
      value: results.issues?.length || 0,
      icon: AlertTriangle,
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      textColor: 'text-red-700',
      iconColor: 'text-red-600',
      valueColor: 'text-red-900'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      {metrics.map(({ label, value, icon: Icon, bgColor, borderColor, textColor, iconColor, valueColor }) => (
        <Card key={label} className={`${bgColor} ${borderColor}`}>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm font-medium ${textColor}`}>{label}</p>
                <p className={`text-3xl font-bold ${valueColor}`}>{value}</p>
              </div>
              <Icon className={`w-8 h-8 ${iconColor}`} />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

const SentimentCard = ({ sentiment }: { sentiment: AnalysisResults['metrics'] }) => {
  if (!sentiment?.sentiment_distribution) return null;

  const colorMap = {
    green: {
      container: 'bg-green-50',
      icon: 'text-green-600',
      text: 'text-green-800',
      value: 'text-green-600',
      subtext: 'text-green-500',
    },
    gray: {
      container: 'bg-gray-50',
      icon: 'bg-gray-400',
      text: 'text-gray-800',
      value: 'text-gray-600',
      subtext: 'text-gray-500',
    },
    red: {
      container: 'bg-red-50',
      icon: 'text-red-600',
      text: 'text-red-800',
      value: 'text-red-600',
      subtext: 'text-red-500',
    },
  };

  const items = [
    { type: 'POSITIVE', count: sentiment.sentiment_distribution.POSITIVE || 0, icon: ThumbsUp, color: 'green' as const },
    { type: 'NEUTRAL', count: sentiment.sentiment_distribution.NEUTRAL || 0, icon: null, color: 'gray' as const },
    { type: 'NEGATIVE', count: sentiment.sentiment_distribution.NEGATIVE || 0, icon: ThumbsDown, color: 'red' as const }
  ];

  const total = Object.values(sentiment.sentiment_distribution).reduce((a, b) => a + b, 0);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sentiment Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {items.map(({ type, count, icon: Icon, color }) => {
            const percentage = total > 0 ? (count / total * 100).toFixed(1) : '0';
            const colors = colorMap[color];
            return (
              <div key={type} className={`flex items-center justify-between p-3 rounded-md ${colors.container}`}>
                <div className="flex items-center gap-2">
                  {Icon ? (
                    <Icon className={`w-4 h-4 ${colors.icon}`} />
                  ) : (
                    <span className={`w-4 h-4 rounded-full ${colors.icon}`}></span>
                  )}
                  <span className={`text-sm font-medium capitalize ${colors.text}`}>
                    {type.toLowerCase()}
                  </span>
                </div>
                <div className="text-right">
                  <p className={`text-2xl font-bold ${colors.value}`}>{count}</p>
                  <p className={`text-sm ${colors.subtext}`}>{percentage}%</p>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
};

function FeatureAnalyzer() {
  // State management
  const [htmlInput, setHtmlInput] = useState('');
  const [productType, setProductType] = useState('gaming controller');
  const [targetFeature, setTargetFeature] = useState('');
  const [customFeature, setCustomFeature] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState('');
  const [apiHealth, setApiHealth] = useState<boolean | null>(null);

  // Check API health on component mount
  React.useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        const data = await response.json();
        setApiHealth(data.status === 'healthy' && data.openai_configured);
      } else {
        setApiHealth(false);
      }
    } catch (error) {
      setApiHealth(false);
    }
  };

  const handleAnalyze = useCallback(async () => {
    if (!htmlInput.trim()) {
      setError('Please enter HTML content to analyze.');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      // Determine the target part and whether to use LLM-generated keywords
      const finalTargetPart = targetFeature === 'other' ? customFeature : targetFeature;
      const useLlmKeywords = targetFeature === 'other' && customFeature.trim() !== '';

      const requestData: AnalysisRequest = {
        html: htmlInput,
        product_type: productType,
        target_part: finalTargetPart || undefined,
        use_llm_keywords: useLlmKeywords
      };

      const response = await fetch(`${API_BASE_URL}/analyze-html`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
      }

      const data: AnalysisResults = await response.json();
      setResults(data);
    } catch (err) {
      setError(`Analysis failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }, [htmlInput, productType, targetFeature, customFeature]);

  const handleClear = () => {
    setHtmlInput('');
    setResults(null);
    setError('');
    setTargetFeature('');
    setCustomFeature('');
  };

  const downloadResults = () => {
    if (!results) return;
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analysis_${productType.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <main className="p-4 sm:p-8 max-w-6xl mx-auto">
      {/* Header */}
      <header className="mb-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-blue-600 rounded-lg">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold">AI-Enhanced Review Analyzer</h1>
        </div>
        <p className="text-md sm:text-lg text-gray-600">
          Paste HTML to extract and analyze product reviews for deep insights.
        </p>
        
        {/* API Health Status */}
        <div className="flex items-center gap-2 mt-4">
          <div className={`w-3 h-3 rounded-full ${apiHealth === true ? 'bg-green-500' : apiHealth === false ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'}`} />
          <span className="text-sm text-gray-600">
            {apiHealth === true ? 'API Connected & Ready' : 
             apiHealth === false ? 'API Connection Failed' : 
             'Checking API Status...'}
          </span>
        </div>
      </header>

      {/* Input Configuration */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Analysis Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* HTML Input */}
          <div className="space-y-3">
            <Label htmlFor="html-input" className="text-base font-semibold flex items-center gap-2">
              <Code className="w-5 h-5" />
              Paste HTML Source Code
            </Label>
            <Textarea
              id="html-input"
              value={htmlInput}
              onChange={(e) => setHtmlInput(e.target.value)}
              placeholder="Paste your HTML content containing reviews here..."
              className="min-h-[200px] font-mono text-sm"
            />
          </div>

          {/* Configuration Options */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="product-type">Product Type</Label>
              <Input
                id="product-type"
                value={productType}
                onChange={(e) => setProductType(e.target.value)}
                placeholder="e.g., gaming controller, headphones"
              />
              <p className="text-xs text-gray-500">
                Helps the AI understand context for better analysis.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="target-feature">Focus on a Specific Feature (Optional)</Label>
              <Select value={targetFeature} onValueChange={setTargetFeature}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a feature to focus on" />
                </SelectTrigger>
                <SelectContent>
                  {COMMON_FEATURES.map((feature) => (
                    <SelectItem key={feature.value} value={feature.value}>
                      {feature.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Custom Feature Input */}
          {targetFeature === 'other' && (
            <div className="space-y-2 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <Label htmlFor="custom-feature">Specify Custom Feature</Label>
              <Input
                id="custom-feature"
                value={customFeature}
                onChange={(e) => setCustomFeature(e.target.value)}
                placeholder="e.g., trigger sensitivity, RGB lighting"
              />
              <p className="text-xs text-blue-600">
                💡 AI will generate keywords to filter reviews for your custom feature.
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3">
            <Button
              onClick={handleAnalyze}
              disabled={loading || !apiHealth || !htmlInput.trim()}
              className="flex-1"
              size="lg"
            >
              {loading ? (
                <>
                  <Activity className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Start AI Analysis"
              )}
            </Button>
            <Button onClick={handleClear} variant="outline" size="lg" className="w-full sm:w-auto">
              Clear All
            </Button>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertTriangle className="w-4 h-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Loading State */}
      {loading && (
        <Card className="mb-6">
          <CardContent className="p-8">
            <div className="flex flex-col items-center space-y-4">
              <div className="relative">
                <div className="w-16 h-16 border-4 border-blue-200 rounded-full"></div>
                <div className="w-16 h-16 border-4 border-blue-600 rounded-full border-t-transparent animate-spin absolute top-0"></div>
              </div>
              <div className="text-center space-y-2">
                <h3 className="text-lg font-semibold text-gray-700">AI Analysis in Progress</h3>
                <p className="text-sm text-gray-500">
                  Running sentiment analysis, extracting themes, and generating insights...
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {results && (
        <section className="space-y-6">
          {results.insights?.executive_summary && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lightbulb className="w-5 h-5" />
                  Executive Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg leading-relaxed text-gray-700">
                  {results.insights.executive_summary}
                </p>
              </CardContent>
            </Card>
          )}

          <AnalysisOverviewCard results={results} />

          {results.analysis_metadata && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-5 h-5" />
                  Analysis Details
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-gray-500" />
                    <div>
                      <p className="text-gray-600">Time</p>
                      <p className="font-semibold">{results.analysis_metadata.analysis_time_seconds.toFixed(1)}s</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4 text-gray-500" />
                    <div>
                      <p className="text-gray-600">Tokens</p>
                      <p className="font-semibold">{results.analysis_metadata.token_usage.total_tokens.toLocaleString()}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-gray-500" />
                    <div>
                      <p className="text-gray-600">Est. Cost</p>
                      <p className="font-semibold">${results.analysis_metadata.token_usage.estimated_cost_usd.toFixed(4)}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Target className="w-4 h-4 text-gray-500" />
                    <div>
                      <p className="text-gray-600">Focus</p>
                      <p className="font-semibold capitalize">{formatThemeName(results.analysis_metadata.target_part || 'General')}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {results.metrics && <SentimentCard sentiment={results.metrics} />}

            {results.insights?.key_insights && results.insights.key_insights.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Lightbulb className="w-5 h-5" />
                    Key Insights
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3">
                    {results.insights.key_insights.map((insight, index) => (
                      <li key={index} className="flex items-start gap-3">
                        <Lightbulb className="w-5 h-5 text-blue-600 flex-shrink-0 mt-1" />
                        <p className="text-gray-700">{insight}</p>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}
          </div>
          
          {results.themes?.themes && Object.keys(results.themes.themes).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  Theme Analysis
                  {results.themes.target_part && (
                    <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                      Focus: {formatThemeName(results.themes.target_part)}
                    </span>
                  )}
                </CardTitle>
                {results.themes.filtered_review_count != null && results.themes.original_review_count != null && (
                  <p className="text-sm text-gray-600">
                    Analyzed {results.themes.filtered_review_count} of {results.themes.original_review_count} relevant reviews.
                  </p>
                )}
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {Object.entries(results.themes.themes).map(([themeName, themeData]) => (
                    <article key={themeName} className="border rounded-lg p-4 bg-gray-50">
                      <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-2">
                        <h3 className="text-lg font-semibold text-gray-800">
                          {formatThemeName(themeName)}
                        </h3>
                        <div className="flex items-center gap-2 flex-shrink-0">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getSentimentColor(themeData.sentiment)}`}>
                            {themeData.sentiment}
                          </span>
                          <span className="text-sm text-gray-600">
                            {themeData.mentions} mentions
                          </span>
                        </div>
                      </header>
                      
                      <div className="space-y-4">
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-medium text-gray-700">Confidence:</span>
                            <div className="w-full bg-gray-200 rounded-full h-2.5">
                                <div 
                                  className="bg-blue-500 h-2.5 rounded-full" 
                                  style={{ width: `${themeData.confidence * 100}%` }}
                                />
                            </div>
                            <span className="text-sm font-medium text-gray-600">{(themeData.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>

                        {themeData.example_quote && (
                          <div>
                            <p className="text-sm font-medium text-gray-700 mb-2">Example Quote:</p>
                            <blockquote className="italic text-gray-600 border-l-4 border-gray-300 pl-4 text-sm bg-white p-2 rounded-r">
                              "{themeData.example_quote}"
                            </blockquote>
                          </div>
                        )}

                        {themeData.key_phrases?.length > 0 && (
                          <div>
                            <p className="text-sm font-medium text-gray-700 mb-2">Key Phrases:</p>
                            <div className="flex flex-wrap gap-2">
                              {themeData.key_phrases.map((phrase, index) => (
                                <span key={index} className="px-2 py-1 bg-gray-200 text-gray-800 rounded-md text-xs border border-gray-300">
                                  {formatThemeName(phrase)}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {results.issues && results.issues.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  Issues Identified
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Accordion type="single" collapsible className="w-full">
                  {results.issues.map((issue, index) => (
                    <AccordionItem key={index} value={`issue-${index}`}>
                      <AccordionTrigger className="text-left hover:bg-red-50/50 px-4 py-3 rounded-lg">
                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between w-full pr-4 gap-2">
                          <div className="flex-1">
                            <h4 className="font-semibold text-red-900">
                              {formatThemeName(issue.issue_name)}
                            </h4>
                            <p className="text-sm text-red-800 mt-1">{issue.description}</p>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(issue.severity)}`}>
                              {issue.severity}
                            </span>
                            <span className="text-xs text-gray-600">
                              {issue.frequency} mentions
                            </span>
                          </div>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <div className="space-y-3 pt-3 border-t">
                          <p className="text-sm font-medium text-gray-700">Example Quote:</p>
                          <blockquote className="p-3 bg-gray-100 rounded text-sm italic text-gray-700 border-l-4 border-red-300">
                            "{issue.example_quote}"
                          </blockquote>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              </CardContent>
            </Card>
          )}

          {results.insights?.recommendations && results.insights.recommendations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  AI-Powered Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {results.insights.recommendations.map((rec, index) => (
                    <article key={index} className="border rounded-lg p-4 bg-gradient-to-r from-blue-50 to-indigo-50">
                      <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-3 gap-2">
                        <h3 className="font-semibold text-gray-800 flex-1 pr-4">
                          {rec.recommendation}
                        </h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium border flex-shrink-0 ${getPriorityColor(rec.priority)}`}>
                          {rec.priority} priority
                        </span>
                      </header>
                      
                      <div className="space-y-3 text-sm">
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Rationale:</p>
                          <p className="text-gray-600 leading-relaxed">{rec.rationale}</p>
                        </div>
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Expected Impact:</p>
                          <p className="text-gray-600 leading-relaxed">{rec.impact}</p>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Code className="w-5 h-5" />
                  Raw Analysis Data
                </CardTitle>
                <Button onClick={downloadResults} variant="outline" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Download JSON
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto max-h-96">
                <pre className="text-green-400 text-xs font-mono whitespace-pre-wrap break-all">
                  {JSON.stringify(results, null, 2)}
                </pre>
              </div>
            </CardContent>
          </Card>
        </section>
      )}

      {/* Footer */}
      <footer className="text-center text-sm text-gray-500 space-y-2 mt-12">
        <p>
          Powered by OpenAI & Local Transformer Models
        </p>
        <div className="flex items-center justify-center gap-x-4 gap-y-2 flex-wrap">
          <span>🔍 Sentiment Analysis</span>
          <span>🎯 Theme Extraction</span>
          <span>🚨 Issue Detection</span>
          <span>💡 AI Recommendations</span>
        </div>
      </footer>
    </main>
  );
}

export default FeatureAnalyzer;