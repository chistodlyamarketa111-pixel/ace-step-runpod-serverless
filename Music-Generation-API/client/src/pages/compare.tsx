import { useState, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ChevronDown,
  ChevronUp,
  Music,
  Trophy,
  BarChart3,
  Filter,
  ThumbsUp,
  Equal,
  ArrowLeft,
  Users,
  UserCircle,
  ArrowLeftRight,
} from "lucide-react";
import { Link } from "wouter";

type ComparisonCase = {
  id: string;
  category: string;
  title: string;
  style_prompt: string;
  lyrics: string;
  track_a: string;
  track_b: string;
  ready?: boolean;
};

type VoteValue = "A" | "B" | "tie";

const EVALUATORS = ["Антон", "Сергей", "Денис", "Руслан"] as const;
type Evaluator = (typeof EVALUATORS)[number];

const EVALUATOR_KEYS: Record<Evaluator, string> = {
  "Антон": "anton",
  "Сергей": "sergey",
  "Денис": "denis",
  "Руслан": "ruslan",
};

const EVALUATOR_COLORS: Record<Evaluator, string> = {
  "Антон": "bg-blue-600 hover:bg-blue-700",
  "Сергей": "bg-emerald-600 hover:bg-emerald-700",
  "Денис": "bg-purple-600 hover:bg-purple-700",
  "Руслан": "bg-orange-600 hover:bg-orange-700",
};

const CATEGORIES = [
  { id: "all", label: "All" },
  { id: "street", label: "Street" },
  { id: "melodic", label: "Melodic" },
  { id: "aggressive", label: "Aggressive" },
  { id: "lyrical", label: "Lyrical" },
  { id: "hit", label: "Hit" },
];

const CATEGORY_COLORS: Record<string, string> = {
  street: "bg-orange-500/15 text-orange-700 dark:text-orange-400 border-orange-500/20",
  melodic: "bg-purple-500/15 text-purple-700 dark:text-purple-400 border-purple-500/20",
  aggressive: "bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/20",
  lyrical: "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/20",
  hit: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/20",
};

function getVote(user: Evaluator, caseId: string): VoteValue | null {
  const key = EVALUATOR_KEYS[user];
  const val = localStorage.getItem(`vote_${key}_${caseId}`);
  if (val === "A" || val === "B" || val === "tie") return val;
  return null;
}

function setVote(user: Evaluator, caseId: string, vote: VoteValue) {
  const key = EVALUATOR_KEYS[user];
  localStorage.setItem(`vote_${key}_${caseId}`, vote);
}

type TrackMapping = Record<string, { track_a: string; track_b: string }>;

function UserSelectionScreen({ onSelect }: { onSelect: (user: Evaluator) => void }) {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="max-w-md mx-auto px-4 space-y-8 text-center">
        <div className="space-y-3">
          <Users className="w-12 h-12 mx-auto text-muted-foreground" />
          <h1 className="text-2xl font-bold">Blind Comparison</h1>
          <p className="text-muted-foreground">
            Выберите ваше имя, чтобы начать оценку
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4">
          {EVALUATORS.map((name) => (
            <Button
              key={name}
              size="lg"
              className={`h-16 text-lg font-semibold text-white ${EVALUATOR_COLORS[name]}`}
              onClick={() => onSelect(name)}
            >
              <UserCircle className="w-5 h-5 mr-2" />
              {name}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}

function CombinedResultsTable({
  cases,
  mapping,
  refreshKey,
}: {
  cases: ComparisonCase[];
  mapping: TrackMapping;
  refreshKey: number;
}) {
  const allVotes = useMemo(() => {
    const result: Record<string, Record<Evaluator, VoteValue | null>> = {};
    cases.forEach((c) => {
      const caseVotes: Record<Evaluator, VoteValue | null> = {
        "Антон": null,
        "Сергей": null,
        "Денис": null,
        "Руслан": null,
      };
      EVALUATORS.forEach((user) => {
        caseVotes[user] = getVote(user, c.id);
      });
      result[c.id] = caseVotes;
    });
    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases, refreshKey]);

  const totals = useMemo(() => {
    const t: Record<Evaluator, { aceStep: number; suno: number; tie: number }> = {
      "Антон": { aceStep: 0, suno: 0, tie: 0 },
      "Сергей": { aceStep: 0, suno: 0, tie: 0 },
      "Денис": { aceStep: 0, suno: 0, tie: 0 },
      "Руслан": { aceStep: 0, suno: 0, tie: 0 },
    };
    EVALUATORS.forEach((user) => {
      cases.forEach((c) => {
        const vote = allVotes[c.id]?.[user];
        const m = mapping[c.id];
        if (!vote || !m) return;
        if (vote === "tie") { t[user].tie++; return; }
        const winner = vote === "A" ? m.track_a : m.track_b;
        if (winner === "ace-step") t[user].aceStep++;
        else t[user].suno++;
      });
    });
    return t;
  }, [cases, allVotes, mapping]);

  return (
    <Card className="p-5 space-y-4">
      <div className="flex items-center gap-2 mb-2">
        <Users className="w-5 h-5 text-blue-500" />
        <h3 className="font-semibold text-lg">Combined Results</h3>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-4">
        {EVALUATORS.map((user) => (
          <Card key={user} className="p-3 text-center space-y-1">
            <div className="text-sm font-semibold">{user}</div>
            <div className="text-xs text-muted-foreground space-y-0.5">
              <div className="text-blue-500">ACE: {totals[user].aceStep}</div>
              <div className="text-orange-500">Suno: {totals[user].suno}</div>
              <div className="text-yellow-500">Tie: {totals[user].tie}</div>
            </div>
          </Card>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2 px-2 font-medium">Case</th>
              {EVALUATORS.map((user) => (
                <th key={user} className="text-center py-2 px-2 font-medium">{user}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => {
              const m = mapping[c.id];
              return (
                <tr key={c.id} className="border-b last:border-b-0 hover:bg-muted/50">
                  <td className="py-2 px-2">
                    <span className="font-medium">{c.title}</span>
                    <Badge variant="outline" className={`ml-2 text-xs ${CATEGORY_COLORS[c.category] || ""}`}>
                      {c.category}
                    </Badge>
                  </td>
                  {EVALUATORS.map((user) => {
                    const vote = allVotes[c.id]?.[user];
                    if (!vote) {
                      return (
                        <td key={user} className="text-center py-2 px-2 text-muted-foreground">
                          —
                        </td>
                      );
                    }
                    if (vote === "tie") {
                      return (
                        <td key={user} className="text-center py-2 px-2">
                          <Badge variant="outline" className="bg-yellow-500/15 text-yellow-700 dark:text-yellow-400 border-yellow-500/20">
                            Tie
                          </Badge>
                        </td>
                      );
                    }
                    const winner = m ? (vote === "A" ? m.track_a : m.track_b) : null;
                    const isAceStep = winner === "ace-step";
                    return (
                      <td key={user} className="text-center py-2 px-2">
                        <Badge variant="outline" className={isAceStep
                          ? "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/20"
                          : "bg-orange-500/15 text-orange-700 dark:text-orange-400 border-orange-500/20"
                        }>
                          {isAceStep ? "ACE" : "Suno"}
                        </Badge>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function SummaryBlock({ cases, revealed, mapping, user }: { cases: ComparisonCase[]; revealed: boolean; mapping: TrackMapping | null; user: Evaluator }) {
  const [votes, setVotes] = useState<Record<string, VoteValue | null>>({});

  useEffect(() => {
    const v: Record<string, VoteValue | null> = {};
    cases.forEach((c) => {
      v[c.id] = getVote(user, c.id);
    });
    setVotes(v);
  }, [cases, user]);

  const total = cases.length;
  const voted = Object.values(votes).filter((v) => v !== null).length;
  const aWins = Object.values(votes).filter((v) => v === "A").length;
  const bWins = Object.values(votes).filter((v) => v === "B").length;
  const ties = Object.values(votes).filter((v) => v === "tie").length;

  let aceStepWins = 0;
  let sunoWins = 0;
  let tieCount = 0;
  if (revealed && mapping) {
    cases.forEach((c) => {
      const vote = votes[c.id];
      const m = mapping[c.id];
      if (!vote || !m) return;
      if (vote === "tie") { tieCount++; return; }
      const winner = vote === "A" ? m.track_a : m.track_b;
      if (winner === "ace-step") aceStepWins++;
      else sunoWins++;
    });
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold">{total}</div>
          <div className="text-xs text-muted-foreground mt-1">Total Cases</div>
        </Card>
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-blue-500">{voted}</div>
          <div className="text-xs text-muted-foreground mt-1">Voted</div>
        </Card>
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-emerald-500">{aWins}</div>
          <div className="text-xs text-muted-foreground mt-1">A Wins</div>
        </Card>
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-purple-500">{bWins}</div>
          <div className="text-xs text-muted-foreground mt-1">B Wins</div>
        </Card>
        <Card className="p-4 text-center col-span-2 sm:col-span-1">
          <div className="text-2xl font-bold text-yellow-500">{ties}</div>
          <div className="text-xs text-muted-foreground mt-1">Ties</div>
        </Card>
      </div>
      {revealed && mapping && voted > 0 && (
        <Card className="p-5 space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <Trophy className="w-5 h-5 text-yellow-500" />
            <h3 className="font-semibold text-lg">Results — {user}</h3>
          </div>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-3xl font-bold text-blue-500">{aceStepWins}</div>
              <div className="text-sm font-medium mt-1">ACE-Step</div>
              <div className="text-xs text-muted-foreground">wins</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-yellow-500">{tieCount}</div>
              <div className="text-sm font-medium mt-1">Tie</div>
              <div className="text-xs text-muted-foreground">draws</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-orange-500">{sunoWins}</div>
              <div className="text-sm font-medium mt-1">Suno</div>
              <div className="text-xs text-muted-foreground">wins</div>
            </div>
          </div>
          <div className="w-full h-3 rounded-full bg-muted overflow-hidden flex">
            {voted > 0 && (
              <>
                <div className="h-full bg-blue-500 transition-all" style={{ width: `${(aceStepWins / voted) * 100}%` }} />
                <div className="h-full bg-yellow-500 transition-all" style={{ width: `${(tieCount / voted) * 100}%` }} />
                <div className="h-full bg-orange-500 transition-all" style={{ width: `${(sunoWins / voted) * 100}%` }} />
              </>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}

function CaseCard({
  caseData,
  onVoteChange,
  revealed,
  mapping,
  user,
}: {
  caseData: ComparisonCase;
  onVoteChange: () => void;
  revealed: boolean;
  mapping: TrackMapping | null;
  user: Evaluator;
}) {
  const [expanded, setExpanded] = useState(false);
  const [vote, setVoteState] = useState<VoteValue | null>(() =>
    getVote(user, caseData.id)
  );

  useEffect(() => {
    setVoteState(getVote(user, caseData.id));
  }, [user, caseData.id]);

  const handleVote = (v: VoteValue) => {
    setVote(user, caseData.id, v);
    setVoteState(v);
    onVoteChange();
  };

  const categoryColor =
    CATEGORY_COLORS[caseData.category] ||
    "bg-gray-500/15 text-gray-700 dark:text-gray-400 border-gray-500/20";

  const lyricsLines = caseData.lyrics.split("\n");
  const isLong = lyricsLines.length > 6;
  const displayLyrics = expanded
    ? caseData.lyrics
    : lyricsLines.slice(0, 6).join("\n") + (isLong ? "\n..." : "");

  return (
    <Card className="p-5 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <h3 className="font-semibold text-base">{caseData.title}</h3>
            <Badge variant="outline" className={categoryColor}>
              {caseData.category}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground">{caseData.style_prompt}</p>
        </div>
        <span className="text-xs text-muted-foreground font-mono shrink-0">
          {caseData.id}
        </span>
      </div>

      <div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors mb-2"
        >
          {expanded ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )}
          {expanded ? "Collapse lyrics" : "Show lyrics"}
        </button>
        {(expanded || !isLong) && (
          <pre className="text-sm whitespace-pre-wrap font-sans bg-muted/50 rounded-md p-3 max-h-64 overflow-y-auto">
            {displayLyrics}
          </pre>
        )}
        {!expanded && isLong && (
          <pre className="text-sm whitespace-pre-wrap font-sans bg-muted/50 rounded-md p-3">
            {displayLyrics}
          </pre>
        )}
      </div>

      {caseData.ready === false && (
        <div className="text-center py-4 bg-muted/30 rounded-lg">
          <p className="text-sm text-muted-foreground">Audio not yet generated. Awaiting batch processing.</p>
        </div>
      )}

      {caseData.ready !== false && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Music className="w-4 h-4 text-emerald-500" />
              <span className="text-sm font-medium">Track A</span>
              {revealed && mapping?.[caseData.id] && (
                <Badge variant="outline" className={mapping[caseData.id].track_a === "ace-step" ? "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/20" : "bg-orange-500/15 text-orange-700 dark:text-orange-400 border-orange-500/20"}>
                  {mapping[caseData.id].track_a === "ace-step" ? "ACE-Step" : "Suno"}
                </Badge>
              )}
            </div>
            <audio controls className="w-full" preload="none">
              <source src={caseData.track_a} type="audio/mpeg" />
            </audio>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Music className="w-4 h-4 text-purple-500" />
              <span className="text-sm font-medium">Track B</span>
              {revealed && mapping?.[caseData.id] && (
                <Badge variant="outline" className={mapping[caseData.id].track_b === "ace-step" ? "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/20" : "bg-orange-500/15 text-orange-700 dark:text-orange-400 border-orange-500/20"}>
                  {mapping[caseData.id].track_b === "ace-step" ? "ACE-Step" : "Suno"}
                </Badge>
              )}
            </div>
            <audio controls className="w-full" preload="none">
              <source src={caseData.track_b} type="audio/mpeg" />
            </audio>
          </div>
        </div>
      )}

      <div className="flex items-center gap-3 pt-1">
        <Button
          variant={vote === "A" ? "default" : "outline"}
          size="sm"
          onClick={() => handleVote("A")}
          className={vote === "A" ? "bg-emerald-600 hover:bg-emerald-700" : ""}
        >
          <ThumbsUp className="w-4 h-4 mr-1" />
          A better
        </Button>
        <Button
          variant={vote === "B" ? "default" : "outline"}
          size="sm"
          onClick={() => handleVote("B")}
          className={vote === "B" ? "bg-purple-600 hover:bg-purple-700" : ""}
        >
          <ThumbsUp className="w-4 h-4 mr-1" />
          B better
        </Button>
        <Button
          variant={vote === "tie" ? "default" : "outline"}
          size="sm"
          onClick={() => handleVote("tie")}
          className={vote === "tie" ? "bg-yellow-600 hover:bg-yellow-700" : ""}
        >
          <Equal className="w-4 h-4 mr-1" />
          Tie
        </Button>
        {vote && (
          <span className="text-sm text-muted-foreground ml-2">
            Your vote: <strong>{vote === "tie" ? "Tie" : vote}</strong>
          </span>
        )}
      </div>
    </Card>
  );
}

export default function ComparePage() {
  const [activeCategory, setActiveCategory] = useState("all");
  const [refreshKey, setRefreshKey] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [mapping, setMapping] = useState<TrackMapping | null>(null);
  const [currentUser, setCurrentUser] = useState<Evaluator | null>(() => {
    const saved = localStorage.getItem("compare_current_user");
    if (saved && EVALUATORS.includes(saved as Evaluator)) return saved as Evaluator;
    return null;
  });

  const handleSelectUser = (user: Evaluator) => {
    setCurrentUser(user);
    localStorage.setItem("compare_current_user", user);
    setRevealed(false);
    setMapping(null);
    setRefreshKey((k) => k + 1);
  };

  const handleSwitchUser = () => {
    setCurrentUser(null);
    localStorage.removeItem("compare_current_user");
    setRevealed(false);
    setMapping(null);
  };

  const { data: cases = [], isLoading, error } = useQuery<ComparisonCase[]>({
    queryKey: ["/api/compare/cases"],
  });

  const filteredCases = useMemo(() => {
    if (activeCategory === "all") return cases;
    return cases.filter((c) => c.category === activeCategory);
  }, [cases, activeCategory]);

  const votedCount = useMemo(() => {
    if (!currentUser) return 0;
    return cases.filter((c) => getVote(currentUser, c.id) !== null).length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases, refreshKey, currentUser]);

  const handleVoteChange = () => {
    setRefreshKey((k) => k + 1);
  };

  const handleReveal = async () => {
    try {
      const res = await fetch("/api/compare/reveal");
      const data = await res.json();
      setMapping(data);
      setRevealed(true);
    } catch (e) {
      console.error("Failed to reveal results", e);
    }
  };

  if (!currentUser) {
    return <UserSelectionScreen onSelect={handleSelectUser} />;
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <Link href="/">
                <Button variant="ghost" size="icon" className="shrink-0">
                  <ArrowLeft className="w-4 h-4" />
                </Button>
              </Link>
              <h1 className="text-2xl font-bold flex items-center gap-2">
                <BarChart3 className="w-6 h-6" />
                Blind Comparison
              </h1>
            </div>
            <p className="text-sm text-muted-foreground ml-12">
              Listen to Track A and Track B, then vote which sounds better. Identity is hidden.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-sm py-1 px-3">
              <UserCircle className="w-4 h-4 mr-1" />
              {currentUser}
            </Badge>
            <Button variant="outline" size="sm" onClick={handleSwitchUser}>
              <ArrowLeftRight className="w-4 h-4 mr-1" />
              Switch
            </Button>
          </div>
        </div>

        {!isLoading && cases.length > 0 && (
          <SummaryBlock key={`${refreshKey}-${currentUser}`} cases={cases} revealed={revealed} mapping={mapping} user={currentUser} />
        )}

        {revealed && mapping && cases.length > 0 && (
          <CombinedResultsTable cases={cases} mapping={mapping} refreshKey={refreshKey} />
        )}

        {!revealed && votedCount >= 50 && (
          <Card className="p-5 text-center space-y-3">
            <p className="text-sm text-muted-foreground">
              You've voted on all {votedCount} cases! Ready to see which engine made which track?
            </p>
            <Button size="lg" onClick={handleReveal} className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
              <Trophy className="w-5 h-5 mr-2" />
              Show Results
            </Button>
          </Card>
        )}

        {(revealed || votedCount > 0) && (
          <div className="flex justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const key = EVALUATOR_KEYS[currentUser];
                cases.forEach((c) => localStorage.removeItem(`vote_${key}_${c.id}`));
                setRevealed(false);
                setMapping(null);
                setRefreshKey((k) => k + 1);
              }}
            >
              Restart
            </Button>
          </div>
        )}

        <div className="flex items-center gap-2 flex-wrap">
          <Filter className="w-4 h-4 text-muted-foreground" />
          {CATEGORIES.map((cat) => (
            <Button
              key={cat.id}
              variant={activeCategory === cat.id ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveCategory(cat.id)}
            >
              {cat.label}
              {cat.id !== "all" && (
                <span className="ml-1 text-xs opacity-70">
                  ({cases.filter((c) => c.category === cat.id).length})
                </span>
              )}
            </Button>
          ))}
        </div>

        {isLoading && (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <Card key={i} className="p-5 space-y-4 animate-pulse">
                <div className="h-6 bg-muted rounded w-1/3" />
                <div className="h-4 bg-muted rounded w-2/3" />
                <div className="grid grid-cols-2 gap-4">
                  <div className="h-12 bg-muted rounded" />
                  <div className="h-12 bg-muted rounded" />
                </div>
                <div className="flex gap-3">
                  <div className="h-8 bg-muted rounded w-24" />
                  <div className="h-8 bg-muted rounded w-24" />
                  <div className="h-8 bg-muted rounded w-16" />
                </div>
              </Card>
            ))}
          </div>
        )}

        {error && (
          <Card className="p-6 text-center">
            <p className="text-destructive">Failed to load comparison cases.</p>
            <p className="text-sm text-muted-foreground mt-1">
              Make sure the API is running and cases are configured.
            </p>
          </Card>
        )}

        {!isLoading && !error && filteredCases.length === 0 && (
          <Card className="p-6 text-center">
            <p className="text-muted-foreground">No cases found for this category.</p>
          </Card>
        )}

        <div className="space-y-4">
          {filteredCases.map((c) => (
            <CaseCard key={`${c.id}-${refreshKey}-${currentUser}`} caseData={c} onVoteChange={handleVoteChange} revealed={revealed} mapping={mapping} user={currentUser} />
          ))}
        </div>
      </div>
    </div>
  );
}
