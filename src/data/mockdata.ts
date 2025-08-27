// All interfaces with corrections applied

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


export const mockApiResponse: FullApiResponse = {
  "comparison": {
    "shared_themes": [
      "Customization and Controls",
      "Value and Price",
      "Performance and Responsiveness",
      "Ergonomics and Comfort",
      "Software and Connectivity",
      "Build Quality and Durability",
      "Battery and Charging"
    ],
    "strategic_insights": {
      "areas_to_improve": [
        "Companion app pairing and firmware update reliability (onboarding friction).",
        "Connectivity and power stability (random power downs/disconnections).",
        "Thumbstick durability and wear (grinding, longevity).",
        "Perceived value/price (mixed sentiment on value for money).",
        "Battery and charging experience (mixed sentiment on longevity/charging behavior).",
        "General build quality perceptions (move mixed sentiment toward positive)."
      ],
      "competitive_advantages": [
        "Clear edge in ergonomics and comfort (positive vs mixed), supporting longer, more comfortable play sessions.",
        "No prominent reports of a right-trigger click/obstruction issue, suggesting a cleaner trigger reliability profile vs. competitor.",
        "Parity on performance and responsiveness while matching overall rating, avoiding any performance-based switching incentive.",
        "Issue profile is less centered on unique mechanical faults (e.g., competitor’s right trigger), allowing us to emphasize reliability in that subsystem."
      ],
      "recommendations": [
        {
          "impact": "Reduces setup failures and support load; shifts Software and Connectivity sentiment toward positive; lowers early returns.",
          "priority": "high",
          "recommendation": "Harden firmware update and app pairing flow: implement resumable/atomic OTA with rollback, clearer state messaging, robust error recovery, and an in-app diagnostics tool (BLE signal check, cache reset, driver hints)."
        },
        {
          "impact": "Cuts random power downs/disconnections; improves session reliability; increases CSAT and review sentiment.",
          "priority": "high",
          "recommendation": "Stabilize connectivity and power: add firmware watchdog for brownouts, refine sleep/wake thresholds, debounce power events, optimize BLE stack, and improve RF shielding; validate across common interference scenarios."
        },
        {
          "impact": "Mitigates top wear complaints; reduces RMAs; strengthens Build Quality and Durability sentiment.",
          "priority": "high",
          "recommendation": "Increase thumbstick durability: transition to hall-effect sensors or upgraded bushing materials and dust seals; short term add stricter QC screening, improved lubrication, and provide an in-app calibration utility."
        },
        {
          "impact": "Improves perceived battery life and charging reliability; reduces charging-related tickets.",
          "priority": "medium",
          "recommendation": "Battery and charging improvements: tune battery gauge for accurate reporting and lower idle drain, enforce safe charge timeouts, and qualify higher-quality cells; include a certified low-resistance cable in-box."
        },
        {
          "impact": "Improves Value and Price sentiment without deep discounting; expands addressable audience.",
          "priority": "medium",
          "recommendation": "Value positioning: bundle spare stick caps/carry case, introduce limited-time promos or loyalty credits, and communicate durability upgrades to justify price."
        },
        {
          "impact": "Amplifies a clear competitive win; improves conversion for comfort-sensitive buyers.",
          "priority": "medium",
          "recommendation": "Double down on ergonomic leadership: publish comfort validation (pressure mapping, long-session studies), add size/hand-fit guidance, and highlight reduced fatigue in marketing."
        },
        {
          "impact": "Catches defects earlier; accelerates root-cause analysis; prevents negative review cascades.",
          "priority": "high",
          "recommendation": "Expand reliability QA and field telemetry: pre-ship soak tests for power/connectivity, trigger/stick stress cycles, and opt-in telemetry to surface root causes of disconnects and update failures."
        },
        {
          "impact": "Shortens time-to-resolution; preserves ratings by converting detractors to neutrals/promoters.",
          "priority": "medium",
          "recommendation": "Strengthen post-purchase support: in-app guided troubleshooting for pairing/updates/disconnects, rapid RMA for stick wear, and a maintenance guide to extend lifespan."
        },
        {
          "impact": "Risk reduction and marketing proof point on mechanical reliability.",
          "priority": "low",
          "recommendation": "Proactively audit trigger subsystem even without current issues; define gate tests and public reliability specs to differentiate from competitor’s reported trigger obstruction."
        }
      ]
    },
    "summary_table": "| Theme | Your Product Sentiment | Competitor Product Sentiment | Winner |\n|---|---|---|---|\n| Customization and Controls | mixed | mixed | Tie |\n| Value and Price | mixed | mixed | Tie |\n| Performance and Responsiveness | positive | positive | Tie |\n| Ergonomics and Comfort | positive | mixed | Your Product |\n| Software and Connectivity | mixed | mixed | Tie |\n| Build Quality and Durability | mixed | mixed | Tie |\n| Battery and Charging | mixed | mixed | Tie |",
    "theme_sentiment_comparison": {
      "Battery and Charging": {
        "product_a": "mixed",
        "product_b": "mixed"
      },
      "Build Quality and Durability": {
        "product_a": "mixed",
        "product_b": "mixed"
      },
      "Customization and Controls": {
        "product_a": "mixed",
        "product_b": "mixed"
      },
      "Ergonomics and Comfort": {
        "product_a": "positive",
        "product_b": "mixed"
      },
      "Performance and Responsiveness": {
        "product_a": "positive",
        "product_b": "positive"
      },
      "Software and Connectivity": {
        "product_a": "mixed",
        "product_b": "mixed"
      },
      "Value and Price": {
        "product_a": "mixed",
        "product_b": "mixed"
      }
    },
    "unique_to_product_a": [],
    "unique_to_product_b": []
  },
  "product_a": {
    "analysis_metadata": {
      "analysis_date": "2025-08-27T15:00:58.626180",
      "analysis_time_seconds": 90.68,
      "model_used": "gpt-5-suite",
      "product_type": "Your Product",
      "token_usage": {
        "estimated_cost": 0.00284375,
        "total_tokens": 2275
      },
      "total_reviews": 20
    },
    "insights": {
      "executive_summary": "With an average rating of 4.05 across 20 reviews (15 positive, 3 negative), customers consistently praise competitive performance (microswitch buttons, low input delay, hall‑effect sticks with no drift) and convenient on-controller customization with robust battery life when lights are off (up to ~48 hours). Key pain points are thumbstick grinding caused by anti‑friction rings gouging the stick plastic, finicky app/firmware with occasional random power downs and right‑stick spin, limited remapping (Start/Select/Capture) and non‑swappable stick heights, and heavy battery drain when LEDs are set to 100%.",
      "key_insights": [
        "Performance is a primary purchase driver: users report “best response times” and praise microswitch buttons and hall-effect sticks with no drift, fueling 5-star reviews from competitive players.",
        "Battery perception hinges on LEDs: some achieve ~48 hours with lights off, but 100% brightness drains quickly and brightness adjustments feel ineffective.",
        "Software/connectivity sentiment is mixed: while many report stable multi-device use, a minority cite pairing failures, random power downs, and occasional right-stick spin that disproportionately impact satisfaction.",
        "Build quality is polarized: premium feel, sturdy base, and no drift for many vs. recurring thumbstick grinding due to anti-friction rings and some early-batch button/trigger QA defects.",
        "Customization is powerful (on-controller LCD, profiles, remapping) but constrained by non-remappable Start/Select/Capture and lack of swappable stick heights, pushing some users back to the Elite 2.",
        "Despite a few severe defects, overall sentiment skews positive (10x 5-star, 5x 4-star), suggesting targeted fixes to rings, connectivity, and remapping could convert most remaining detractors."
      ],
      "recommendations": [
        {
          "impact": "High",
          "priority": "High",
          "rationale": "Reports of the anti-friction rings gouging plastic within hours cause gritty rotation and returns; switching to a low-wear polymer (e.g., PTFE/POM) or coated ring plus a radiused stick edge and adding a dust wiper, validated by a 200-hour edge-load tribology test, will eliminate the thumbstick grinding issue.",
          "recommendation": "Redesign the thumbstick/anti-friction interface (material and geometry) and validate with accelerated wear testing."
        },
        {
          "impact": "High",
          "priority": "High",
          "rationale": "Users cite finicky app behavior, pairing failures, intermittent disconnects, and anomalous right-stick spin; improving BLE/2.4GHz handshake reliability, telemetry-driven crash logs, and recovery routines will reduce the small but severe failures driving negative reviews.",
          "recommendation": "Stabilize connectivity and app/firmware: fix pairing flows, add auto-reconnect, and implement a watchdog to prevent random power-downs/right-stick spin."
        },
        {
          "impact": "Medium-High",
          "priority": "Medium-High",
          "rationale": "Customers want to remap Start/Select/Capture and adjust stick heights; enabling remap in firmware and offering an optional swappable stick cap/height kit will close gaps versus Elite controllers and retain competitive users.",
          "recommendation": "Unlock full remapping and expand hardware customization options."
        },
        {
          "impact": "Medium",
          "priority": "Medium",
          "rationale": "Battery life is strong with lights off (up to ~48 hours) but drops quickly at 100% brightness; better power profiles and transparent time-remaining guidance align expectations and improve perceived battery performance.",
          "recommendation": "Introduce power-optimized LED profiles and clearer brightness steps, with a lower default LED setting and runtime estimates on the LCD."
        },
        {
          "impact": "Medium",
          "priority": "Medium",
          "rationale": "Early-batch QA issues (sticking A, shoulder failures, abnormal trigger click) hurt durability sentiment; additional inspection gates and statistical sampling will reduce defect leakage and warranty replacements.",
          "recommendation": "Tighten mechanical QA on inputs (A/shoulder/trigger) with end-of-line actuation-force and click-consistency checks."
        }
      ]
    },
    "issues": [
      {
        "description": "Users report the thumbsticks grind against the outer friction ring, causing gritty feel and premature wear.",
        "example_quote": "the thumb sticks grind against the terrible friction ring.",
        "frequency": 2,
        "issue_name": "Thumbstick Grinding and Wear",
        "severity": "high"
      },
      {
        "description": "The controller intermittently powers off and disconnects during use, disrupting gameplay.",
        "example_quote": "random power downs mid-use (every few minutes), random disconnections (which required me to repair every few minutes)",
        "frequency": 1,
        "issue_name": "Random Power Downs and Disconnections",
        "severity": "high"
      },
      {
        "description": "Some users cannot pair the controller with the mobile app, preventing firmware updates across devices.",
        "example_quote": "inability to pair with the mobile app (preventing me from updating it, even after trying on multiple devices)",
        "frequency": 1,
        "issue_name": "App Pairing and Firmware Update Failure",
        "severity": "medium"
      },
      {
        "description": "Certain buttons cannot be remapped and there is no option to change thumbstick heights, limiting customization.",
        "example_quote": "You cannot remap the start select or capture buttons.",
        "frequency": 1,
        "issue_name": "Limited Button Remapping and Customization",
        "severity": "medium"
      },
      {
        "description": "Battery performance appears overstated, with some users reporting only a day of use on full power.",
        "example_quote": "the battery life is an exaggeration.",
        "frequency": 1,
        "issue_name": "Battery Life Shorter Than Expected",
        "severity": "medium"
      }
    ],
    "metrics": {
      "analysis_quality": "high",
      "average_rating": 4.05,
      "average_review_length": 567,
      "rating_distribution": {
        "1.0": 1,
        "2.0": 2,
        "3.0": 2,
        "4.0": 5,
        "5.0": 10
      },
      "sentiment_distribution": {
        "negative": 3,
        "neutral": 2,
        "positive": 15
      },
      "total_reviews": 20
    },
    "themes": {
      "Battery and Charging": {
        "confidence": 0.84,
        "negative_points": [
          "Battery life shorter than advertised with lights at 100%",
          "Lights drain battery significantly; brightness changes have minimal effect",
          "Some report about a day of use in full power mode"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Good to excellent battery life for many users",
          "Up to ~48 hours reported with lights off, exceeding advertised claims",
          "Charging base/stand works reliably and indicates full charge",
          "Fast drop-in charging via USB-C base"
        ],
        "representative_quote": "Lights on 100% will also drain the battery pretty quick... if you turn them off you will get over 30 hours. I have gotten a solid 48 hours."
      },
      "Build Quality and Durability": {
        "confidence": 0.86,
        "negative_points": [
          "Anti-friction rings gouge thumbstick plastic causing grinding and wear",
          "Early batch QA issues (A button, shoulder buttons) and a trigger with abnormal click",
          "Left stick wear grooves within months for some users"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Premium feel for controller and case",
          "No stick drift reported after long-term use by multiple users",
          "Strong magnetic/charging base feels sturdy",
          "Responsive customer support with swift replacement"
        ],
        "representative_quote": "The anti friction rings... after only a few hours of gameplay, the plastic on the thumb stick have gouges in them... They feel like they are grinding when you rotate the stick."
      },
      "Customization and Controls": {
        "confidence": 0.9,
        "negative_points": [
          "Start/Select/Capture buttons cannot be remapped",
          "Thumbstick heights not swappable like some competitors",
          "Menu button placement too close to the stick for some users",
          "Back button spacing not comfortable for some"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Extensive remapping including back buttons and trigger/stick response tuning",
          "On-controller LCD makes on-the-fly edits and profiles easy",
          "Multiple profiles (up to ten) and lighting customization",
          "Microswitch buttons feel like mouse clicks for fast response"
        ],
        "representative_quote": "You cannot remap the start select or capture buttons."
      },
      "Ergonomics and Comfort": {
        "confidence": 0.78,
        "negative_points": [
          "Menu button placement is too close to the left stick for some",
          "Back buttons too close together for some users’ preferences"
        ],
        "overall_sentiment": "positive",
        "positive_points": [
          "Feels great in the hands with comfortable weight",
          "Many users like the bottom/back button layout once learned",
          "Controller feels lighter in a good way compared to other pro controllers",
          "Grip and overall hand feel praised by multiple users"
        ],
        "representative_quote": "Feels great in the hands and doesn't take long getting used to the bottom buttons."
      },
      "Performance and Responsiveness": {
        "confidence": 0.82,
        "negative_points": [
          "One user reports the controller doesn't feel as responsive as Elite Series 2",
          "Right stick spinning during matches for one user",
          "Thumbstick grinding negatively impacts gameplay smoothness for some"
        ],
        "overall_sentiment": "positive",
        "positive_points": [
          "Exceptional response times and low input delay suited for competitive play",
          "Microswitch buttons are highly responsive",
          "Hall-effect sticks praised with no drift for many users",
          "Trigger lock functions help in FPS titles"
        ],
        "representative_quote": "This controller has the best response times I have ever witnessed in a controller."
      },
      "Software and Connectivity": {
        "confidence": 0.85,
        "negative_points": [
          "Buggy/finicky app on Android and Xbox; pairing issues for some",
          "Random power downs and disconnections for at least one user",
          "Occasional in-match connectivity hiccups (e.g., right stick spinning)",
          "Setup can take time and patience"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Connects well and easy to swap between devices with USB dongle",
          "Profiles stored on-controller; companion app can simplify updates",
          "Stable connection reported across a whole house for some users",
          "Base acts as wireless transmitter for fast connection"
        ],
        "representative_quote": "Random power downs mid-use... random disconnections... inability to pair with the mobile app."
      },
      "Value and Price": {
        "confidence": 0.62,
        "negative_points": [
          "Price feels steep for at least one user given the software issues"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Considered better value than competing premium controllers by some",
          "Open-box purchase cited as a good money saver"
        ],
        "representative_quote": "For the price, I’d say it’s a bit steep considering what you’re paying for but other than that it’s not a terrible product."
      }
    }
  },
  "product_b": {
    "analysis_metadata": {
      "analysis_date": "2025-08-27T15:02:17.746939",
      "analysis_time_seconds": 79.12,
      "model_used": "gpt-5-suite",
      "product_type": "Competitor Product",
      "token_usage": {
        "estimated_cost": 0.00547625,
        "total_tokens": 4381
      },
      "total_reviews": 20
    },
    "insights": {
      "executive_summary": "Across 20 reviews (avg 4.05/5; 15 positive, 3 negative), users praise best-in-class response times, micro-switch “mouse-click” buttons, drift-free hall-effect sticks, and convenient on-controller customization with a magnetic charging base. However, multiple durability complaints cite thumbstick grinding where anti-friction rings gouge the stick plastic, while software/connectivity issues include a buggy app, random power downs/disconnections, and occasional right-stick spin; RGB lighting also drains the battery quickly versus advertised figures when left at 100%. Additional friction points include limited remapping (start/select/capture), lack of swappable stick heights, and menu button placement too close to the left stick, with some perceiving the price as steep.",
      "key_insights": [
        "Performance is a core differentiator: users highlight exceptionally low input delay and micro-switch buttons that feel like mouse clicks.",
        "Durability feedback is polarized: premium feel and drift-free hall-effect sticks are praised, but anti-friction rings gouging stick plastic creates grinding after only hours for some users.",
        "Software/connectivity is inconsistent: many report easy multi-device use and profile portability, while others see a buggy app, pairing failures, random power downs/disconnections, and sporadic right-stick spin.",
        "Battery life is context-dependent: 30–48 hours is achievable with lights off, but 100% RGB brightness significantly reduces runtime below advertised figures.",
        "Customization breadth is strong via on-controller screen and profiles, yet gaps remain—start/select/capture cannot be remapped and stick heights aren’t swappable.",
        "Ergonomics are generally praised (lighter weight, usable four back buttons), but menu button placement near the left stick and some preference for Elite Series 2 feel cause defections."
      ],
      "recommendations": [
        {
          "impact": "High",
          "priority": "High",
          "rationale": "Directly addresses reports of thumbstick gouging and grinding and early-batch button defects, which degrade gameplay and drive returns/warranty costs.",
          "recommendation": "Redesign the thumbstick/anti-friction interface (e.g., switch to POM/PTFE or metal ring with low-friction coating and tougher stick skirts), and add focused lifecycle/abrasion testing; tighten hardware QA on shoulder/trigger assemblies."
        },
        {
          "impact": "High",
          "priority": "High",
          "rationale": "Connectivity and app reliability issues undermine otherwise strong performance; resolving them protects the competitive-play value proposition and reduces negative reviews.",
          "recommendation": "Stabilize firmware and companion app: fix random power downs/disconnections and pairing failures, add robust OTA/update fallbacks, and implement telemetry to detect input anomalies (e.g., right-stick spin)."
        },
        {
          "impact": "High",
          "priority": "Medium",
          "rationale": "Removes key customization gaps and addresses button placement pain points cited by competitive users who value precise control layouts.",
          "recommendation": "Expand controls and ergonomics in the next revision: enable remapping of start/select/capture buttons, offer swappable/height-adjustable stick caps, and move the left menu button further from the stick."
        },
        {
          "impact": "Medium-High",
          "priority": "Medium",
          "rationale": "Users report fast drain at 100% brightness and confusion versus the 30-hour claim; better defaults and clearer specs improve satisfaction and trust.",
          "recommendation": "Optimize power and lighting: introduce an aggressive low-power RGB profile by default, expand brightness range with meaningful steps, surface real-time battery estimates, and update marketing to specify hours with lights on vs off."
        },
        {
          "impact": "Medium",
          "priority": "Low",
          "rationale": "Addresses price-sensitivity while preserving the core performance advantages that drive positive sentiment.",
          "recommendation": "Offer a lower-cost 'Lite' SKU (reduced RGB/accessories) and extend/open-box/refurb programs with clear warranty."
        }
      ]
    },
    "issues": [
      {
        "description": "Users report gritty grinding against the friction ring, accelerated wear creating grooves, and calibration issues with the sticks.",
        "example_quote": "No way to change heights and the thumb sticks grind against the terrible friction ring.",
        "frequency": 3,
        "issue_name": "Thumbstick Performance and Durability Problems",
        "severity": "high"
      },
      {
        "description": "Controllers randomly power down or disconnect during use and fail to pair with the companion app.",
        "example_quote": "random power downs mid-use (every few minutes), random disconnections (which required me to repair every few minutes), inability to pair with the mobile app",
        "frequency": 1,
        "issue_name": "Connectivity and Power Instability",
        "severity": "high"
      },
      {
        "description": "The right trigger exhibits a strange click as if obstructed, preventing a full press.",
        "example_quote": "right trigger with a strange click as if there was debris stuck inside not allowing a full click",
        "frequency": 1,
        "issue_name": "Right Trigger Click/Obstruction",
        "severity": "medium"
      },
      {
        "description": "Not all buttons are remappable and there is no option to change thumbstick height.",
        "example_quote": "You cannot remap the start select or capture buttons.",
        "frequency": 1,
        "issue_name": "Limited Remapping and Customization",
        "severity": "medium"
      },
      {
        "description": "Battery duration is perceived as overstated, lasting only about a day in full power mode.",
        "example_quote": "the battery life is an exaggeration... If I leave it full power mode it lasts me the day.",
        "frequency": 1,
        "issue_name": "Battery Life Shorter Than Advertised",
        "severity": "medium"
      }
    ],
    "metrics": {
      "analysis_quality": "high",
      "average_rating": 4.05,
      "average_review_length": 567,
      "rating_distribution": {
        "1.0": 1,
        "2.0": 2,
        "3.0": 2,
        "4.0": 5,
        "5.0": 10
      },
      "sentiment_distribution": {
        "negative": 3,
        "neutral": 2,
        "positive": 15
      },
      "total_reviews": 20
    },
    "themes": {
      "Battery and Charging": {
        "confidence": 0.88,
        "negative_points": [
          "RGB lighting significantly drains the battery",
          "Battery life can be shorter than advertised in high-power modes"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Good overall battery life reported by several users",
          "Magnetic drop-in fast charging base is convenient and strong",
          "Battery indicator and long life achievable with lights off (30–48 hours reported)"
        ],
        "representative_quote": "Lights on 100% will also drain the battery pretty quick... if you turn them off you will get over 30 hours."
      },
      "Build Quality and Durability": {
        "confidence": 0.86,
        "negative_points": [
          "Anti-friction rings gouge the thumbstick plastic causing grinding and reduced smoothness",
          "Early batch QA issues like shoulder buttons failing and an A button defect",
          "Individual defects such as a right trigger click/obstruction and thumbstick ring wear over time"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Premium hardware feel and sturdy construction including the case",
          "Hall effect sticks reported drift-free even after long-term use",
          "Magnetic charging stand feels solid and secure"
        ],
        "representative_quote": "after only a few hours of gameplay, the plastic on the thumb stick have gouges in them from rubbing on the anti friction rings."
      },
      "Customization and Controls": {
        "confidence": 0.9,
        "negative_points": [
          "Not all buttons are remappable (start/select/capture buttons cannot be remapped)",
          "No option to swap or change thumbstick heights",
          "Menu button placement is too close to the stick for some users"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Extensive customization via on-controller screen and app, including profiles, lighting, and response curves",
          "Remapping of back buttons and adjustable stick/trigger responses",
          "Bottom buttons enable advanced control setups and can be mastered for competitive play"
        ],
        "representative_quote": "From the menu you can also customise lighting and many other features like remapping the back buttons, setting response of the analog sticks and triggers."
      },
      "Ergonomics and Comfort": {
        "confidence": 0.8,
        "negative_points": [
          "Menu/start button placement is too close to the left stick",
          "Some users prefer the feel of the Elite Series 2 and reverted to it",
          "Back button spacing not ideal for everyone"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Feels great in the hands with a comfortable, lighter weight",
          "Back button layout is a game-changer for some, enabling full use of four paddles",
          "Overall shape and in-hand feel widely praised"
        ],
        "representative_quote": "Feels great in the hands and doesnt take long getting used to the bottom buttons."
      },
      "Performance and Responsiveness": {
        "confidence": 0.82,
        "negative_points": [
          "Some users feel it doesn't respond as well as the Elite Series 2",
          "Hardware/software issues like a right trigger obstruction or stick grinding can impact gameplay"
        ],
        "overall_sentiment": "positive",
        "positive_points": [
          "Exceptional response times and low input delay praised by competitive players",
          "Micro-switch buttons feel like mouse clicks and are highly responsive",
          "Hall effect sticks and zero dead zone settings deliver precise control"
        ],
        "representative_quote": "This controller has the best response times I have ever witnessed in a controller."
      },
      "Software and Connectivity": {
        "confidence": 0.84,
        "negative_points": [
          "Companion app is buggy/finicky on Android and Xbox",
          "Random power downs and disconnections, with pairing failures to the app for some",
          "Occasional connectivity hiccups causing in-game input anomalies"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Connects well for many users and can easily swap between devices",
          "Profiles stored on the controller simplify switching and setup",
          "Setup guidance helped some users avoid connection issues"
        ],
        "representative_quote": "random power downs mid-use, random disconnections, inability to pair with the mobile app"
      },
      "Value and Price": {
        "confidence": 0.63,
        "negative_points": [
          "Price is viewed as steep by some given the software and durability concerns"
        ],
        "overall_sentiment": "mixed",
        "positive_points": [
          "Considered better value than some premium competitors by several users",
          "Open-box purchases offered meaningful savings"
        ],
        "representative_quote": "For the price, I’d say it’s a bit steep considering what you’re paying for but other than that it’s not a terrible product."
      }
    }
  }
};