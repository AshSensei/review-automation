import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
  NavigationMenuLink,
} from "@/components/ui/navigation-menu";
import { useNavigate } from "react-router-dom";
function Home() {
  const navigate = useNavigate();
  return (
    <div className="p-8 max-w-3xl mx-auto">
      <NavigationMenu className="mb-6">
        <NavigationMenuList>
          <NavigationMenuItem>
            <NavigationMenuLink
              asChild
              onClick={() => navigate("/")}
              className="cursor-pointer"
            >
              <a>Home</a>
            </NavigationMenuLink>
          </NavigationMenuItem>
          <NavigationMenuItem>
            <NavigationMenuLink
              asChild
              onClick={() => navigate("/html-analyzer")}
              className="cursor-pointer"
            >
              <a>HTML Analyzer</a>
            </NavigationMenuLink>
          </NavigationMenuItem>
          <NavigationMenuItem>
            <NavigationMenuLink
              asChild
              onClick={() => navigate("/competitor-analyzer")}
              className="cursor-pointer"
            >
              <a>Competitor Analyzer</a>
            </NavigationMenuLink>
          </NavigationMenuItem>
        </NavigationMenuList>
      </NavigationMenu>

      <h1 className="text-4xl font-bold mb-4">Review Analyzer</h1>

      <Card>
        <CardHeader>
          <CardTitle>Manual HTML Analyzer</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4">
            Copy and paste your HTML to analyze the reviews and get insights
          </p>
          <Button onClick={() => navigate("/html-analyzer")}>
            Open HTML Analyzer
          </Button>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Competitor Analyzer</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4">
            Compare your product with a competitor's product by cross analyzing
            reviews.
          </p>
          <Button onClick={() => navigate("/competitor-analyzer")}>
            Open Competitor Analyzer
          </Button>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Feature Analyzer</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4">
            Analyze a specific feature of your product through analyzing reviews.
          </p>
          <Button onClick={() => navigate("/feature-analyzer")}>
            Open Feature Analyzer
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

export default Home;
