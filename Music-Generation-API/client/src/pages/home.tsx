import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage, FormDescription } from "@/components/ui/form";
import { useToast } from "@/hooks/use-toast";
import type { Job, Comparison } from "@shared/schema";
import {
  Play,
  Square,
  Copy,
  Check,
  Send,
  Clock,
  Music,
  Zap,
  FileJson,
  RefreshCw,
  Terminal,
  BookOpen,
  History,
  AlertCircle,
  CheckCircle2,
  Loader2,
  ExternalLink,
  Download,
  Settings,
  ChevronDown,
  ChevronUp,
  Mic,
  Guitar,
  Shield,
  Server,
  Cpu,
  Activity,
  HardDrive,
  Power,
  RotateCcw,
  Sparkles,
  Wand2,
} from "lucide-react";

const STYLES = [
  "Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip Hop",
  "R&B", "Country", "Folk", "Blues", "Metal", "Ambient",
  "Lo-fi", "Cinematic", "Orchestral", "Synthwave",
];

const INSTRUMENTS = [
  "Piano", "Guitar", "Drums", "Bass", "Violin", "Synthesizer",
  "Flute", "Saxophone", "Trumpet", "Cello", "Harp", "Organ",
];

const KEY_SCALES = [
  "C Major", "C Minor", "C# Major", "C# Minor",
  "D Major", "D Minor", "Eb Major", "Eb Minor",
  "E Major", "E Minor", "F Major", "F Minor",
  "F# Major", "F# Minor", "G Major", "G Minor",
  "Ab Major", "Ab Minor", "A Major", "A Minor",
  "Bb Major", "Bb Minor", "B Major", "B Minor",
];

const TIME_SIGNATURES = [
  { value: "4", label: "4/4 (Common)" },
  { value: "3", label: "3/4 (Waltz)" },
  { value: "2", label: "2/4 (March)" },
  { value: "6", label: "6/8 (Compound)" },
];

const VOCAL_LANGUAGES = [
  { value: "en", label: "English" },
  { value: "zh", label: "Chinese" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "ru", label: "Russian" },
];

const playgroundSchema = z.object({
  engine: z.string(),
  prompt: z.string().min(1, "Prompt is required"),
  lyrics: z.string().optional(),
  duration: z.number().min(10).max(600),
  style: z.string().optional(),
  instrument: z.string().optional(),
  tags: z.string().optional(),
  negative_tags: z.string().optional(),
  title: z.string().optional(),
  bpm: z.number().min(30).max(300).optional(),
  key_scale: z.string().optional(),
  time_signature: z.string().optional(),
  vocal_language: z.string().optional(),
  inference_steps: z.number().min(1).max(200),
  guidance_scale: z.number().min(0).max(30),
  thinking: z.boolean(),
  shift: z.number().min(1).max(5),
  infer_method: z.string(),
  use_adg: z.boolean(),
  seed: z.number().int(),
  hm_temperature: z.number().min(0.1).max(2.0).optional().default(1.0),
  hm_cfg_scale: z.number().min(0).max(10).optional().default(1.5),
  hm_topk: z.number().int().min(1).max(500).optional().default(50),
}).superRefine((data, ctx) => {
  if (data.engine === "heartmula" && data.duration > 300) {
    ctx.addIssue({
      code: z.ZodIssueCode.too_big,
      maximum: 300,
      type: "number",
      inclusive: true,
      message: "HeartMuLa max duration is 300 seconds (5 minutes)",
      path: ["duration"],
    });
  }
});

type PlaygroundFormValues = z.infer<typeof playgroundSchema>;

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; icon: typeof Clock; label: string }> = {
    PENDING: { variant: "secondary", icon: Clock, label: "Pending" },
    IN_QUEUE: { variant: "secondary", icon: Clock, label: "In Queue" },
    IN_PROGRESS: { variant: "default", icon: Loader2, label: "Processing" },
    COMPLETED: { variant: "outline", icon: CheckCircle2, label: "Completed" },
    FAILED: { variant: "destructive", icon: AlertCircle, label: "Failed" },
    CANCELLED: { variant: "destructive", icon: Square, label: "Cancelled" },
  };
  const c = config[status] || config.PENDING;
  const Icon = c.icon;
  return (
    <Badge variant={c.variant} data-testid={`badge-status-${status.toLowerCase()}`}>
      <Icon className={`w-3 h-3 mr-1 ${status === "IN_PROGRESS" ? "animate-spin" : ""}`} />
      {c.label}
    </Badge>
  );
}

function EngineBadge({ engine }: { engine: string }) {
  if (engine === "heartmula") {
    return (
      <Badge variant="outline" className="text-xs bg-pink-500/10 text-pink-700 dark:text-pink-400 border-pink-500/20">
        <Mic className="w-3 h-3 mr-1" />
        HeartMuLa
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-500/20">
      <Guitar className="w-3 h-3 mr-1" />
      ACE-Step
    </Badge>
  );
}

function CodeBlock({ code, language = "json" }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="relative group">
      <pre className="bg-muted rounded-md p-4 overflow-x-auto font-mono text-sm leading-relaxed">
        <code>{code}</code>
      </pre>
      <Button
        size="icon"
        variant="ghost"
        className="absolute top-2 right-2 visibility-hidden group-hover:visibility-visible"
        style={{ visibility: "hidden" }}
        onMouseEnter={(e) => { e.currentTarget.style.visibility = "visible"; }}
        onClick={handleCopy}
        data-testid="button-copy-code"
      >
        {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
      </Button>
      <div
        className="absolute top-0 right-0 w-16 h-12"
        onMouseEnter={(e) => {
          const btn = e.currentTarget.nextElementSibling?.previousElementSibling as HTMLElement;
          if (btn) btn.style.visibility = "visible";
        }}
      />
    </div>
  );
}

function EndpointCard({
  method,
  path,
  description,
}: {
  method: string;
  path: string;
  description: string;
}) {
  const methodColor =
    method === "POST"
      ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
      : "bg-blue-500/15 text-blue-700 dark:text-blue-400";
  return (
    <div className="flex items-start gap-3 p-3 rounded-md hover-elevate">
      <span
        className={`${methodColor} font-mono text-xs font-bold px-2 py-1 rounded-md shrink-0`}
        data-testid={`text-method-${method.toLowerCase()}-${path.replace(/[/:]/g, "-")}`}
      >
        {method}
      </span>
      <div className="min-w-0">
        <code className="font-mono text-sm text-foreground" data-testid="text-endpoint-path">{path}</code>
        <p className="text-sm text-muted-foreground mt-0.5">{description}</p>
      </div>
    </div>
  );
}

function ApiDocsTab() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-1">Authentication</h3>
        <p className="text-sm text-muted-foreground mb-3">
          All API requests are processed through this service. No additional authentication is required for the endpoints below.
        </p>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Endpoints</h3>
        <Card className="divide-y divide-border">
          <EndpointCard
            method="POST"
            path="/api/generate"
            description="Submit a new music generation job. Choose engine: ace-step or heartmula."
          />
          <EndpointCard
            method="GET"
            path="/api/jobs/:id"
            description="Check the status and retrieve details of a specific generation job."
          />
          <EndpointCard
            method="GET"
            path="/api/jobs"
            description="List all generation jobs with their current status."
          />
          <EndpointCard
            method="GET"
            path="/api/jobs/:id/audio"
            description="Download the generated audio file for a completed job."
          />
        </Card>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Engines</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Guitar className="w-4 h-4 text-blue-500" />
              <span className="font-semibold text-sm">ACE-Step v1.5</span>
            </div>
            <p className="text-xs text-muted-foreground mb-2">
              Instrumental & vocal music generation with fine-grained DiT controls.
              Best for short clips (15-120s) with precise parameter tuning.
            </p>
            <div className="text-xs text-muted-foreground">
              Params: inference_steps, guidance_scale, bpm, key_scale, shift, thinking, use_adg
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Mic className="w-4 h-4 text-pink-500" />
              <span className="font-semibold text-sm">HeartMuLa</span>
            </div>
            <p className="text-xs text-muted-foreground mb-2">
              Full song generation with vocals and lyrics, up to 5 minutes.
              LLM-based model producing studio-quality complete songs.
            </p>
            <div className="text-xs text-muted-foreground">
              Params: tags, negative_tags, lyrics, title, duration, seed, temperature, cfg_scale, topk
            </div>
          </Card>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Request Example (ACE-Step)</h3>
        <CodeBlock
          code={JSON.stringify(
            {
              engine: "ace-step",
              prompt: "Upbeat indie pop with jangly guitars and energetic vocals",
              lyrics: "[Verse 1]\nWalking down the street\nMusic in my feet",
              duration: 60,
              style: "Pop",
              instrument: "Guitar",
              bpm: 120,
              inference_steps: 40,
              guidance_scale: 7,
            },
            null,
            2
          )}
        />
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Request Example (HeartMuLa)</h3>
        <CodeBlock
          code={JSON.stringify(
            {
              engine: "heartmula",
              prompt: "A dreamy love ballad with emotional female vocals",
              lyrics: "[Verse 1]\nUnder the starlight\nI found you tonight\n\n[Chorus]\nYou are my everything",
              duration: 180,
              tags: "Pop ballad, emotional, female vocals, piano",
              negative_tags: "harsh, distortion, noise",
              title: "Starlight",
              temperature: 1.0,
              cfg_scale: 1.5,
              topk: 50,
              seed: -1,
            },
            null,
            2
          )}
        />
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Parameters</h3>
        <div className="space-y-1">
          {[
            { name: "engine", type: "string", required: false, desc: "Engine: 'ace-step' (default) or 'heartmula'" },
            { name: "prompt", type: "string", required: true, desc: "Text description of the music style, genre, instruments, mood" },
            { name: "lyrics", type: "string", required: false, desc: "Lyrics with structure tags: [Verse], [Chorus], [Bridge], etc." },
            { name: "duration", type: "number", required: false, desc: "Duration in seconds (10-360 HeartMuLa, 10-600 ACE-Step, default: 30)" },
            { name: "style", type: "string", required: false, desc: "Musical style/genre" },
            { name: "instrument", type: "string", required: false, desc: "Primary instrument (ACE-Step only)" },
            { name: "tags", type: "string", required: false, desc: "Style tags, comma-separated (HeartMuLa)" },
            { name: "negative_tags", type: "string", required: false, desc: "Elements to exclude from generation (HeartMuLa)" },
            { name: "title", type: "string", required: false, desc: "Track title (HeartMuLa)" },
            { name: "temperature", type: "number", required: false, desc: "Creativity control, 0.1-2.0 (HeartMuLa, default: 1.0)" },
            { name: "cfg_scale", type: "number", required: false, desc: "Prompt adherence, 0-10 (HeartMuLa, default: 1.5)" },
            { name: "topk", type: "number", required: false, desc: "Top-K sampling, 1-500 (HeartMuLa, default: 50)" },
            { name: "bpm", type: "number", required: false, desc: "Tempo in BPM, 30-300 (ACE-Step only)" },
            { name: "key_scale", type: "string", required: false, desc: "Key and scale (ACE-Step only)" },
            { name: "inference_steps", type: "number", required: false, desc: "Inference steps, 1-200 (ACE-Step only)" },
            { name: "guidance_scale", type: "number", required: false, desc: "Prompt guidance, 0-30 (ACE-Step only)" },
            { name: "thinking", type: "boolean", required: false, desc: "LM thinking mode (ACE-Step only)" },
            { name: "shift", type: "number", required: false, desc: "Timestep shift, 1-5 (ACE-Step only)" },
            { name: "infer_method", type: "string", required: false, desc: "'ode' or 'sde' (ACE-Step only)" },
            { name: "use_adg", type: "boolean", required: false, desc: "Adaptive Dual Guidance (ACE-Step only)" },
            { name: "seed", type: "number", required: false, desc: "Fixed seed for reproducibility (-1 for random)" },
          ].map((p) => (
            <div key={p.name} className="flex items-start gap-3 p-2.5 rounded-md flex-wrap">
              <code className="font-mono text-sm shrink-0">{p.name}</code>
              <Badge variant="outline" className="shrink-0 text-xs">{p.type}</Badge>
              {p.required && (
                <Badge variant="default" className="shrink-0 text-xs">required</Badge>
              )}
              <span className="text-sm text-muted-foreground">{p.desc}</span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Status Values</h3>
        <div className="flex flex-wrap gap-2">
          {["PENDING", "IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED"].map((s) => (
            <StatusBadge key={s} status={s} />
          ))}
        </div>
      </div>
    </div>
  );
}

function PlaygroundTab() {
  const { toast } = useToast();
  const [response, setResponse] = useState<string | null>(null);
  const [showAdvancedAceStep, setShowAdvancedAceStep] = useState(false);
  const [showAdvancedHeartMuLa, setShowAdvancedHeartMuLa] = useState(false);

  const form = useForm<PlaygroundFormValues>({
    resolver: zodResolver(playgroundSchema),
    defaultValues: {
      engine: "ace-step",
      prompt: "",
      lyrics: "",
      duration: 30,
      style: "",
      instrument: "",
      tags: "",
      negative_tags: "",
      title: "",
      bpm: undefined,
      key_scale: "",
      time_signature: "",
      vocal_language: "",
      inference_steps: 40,
      guidance_scale: 7,
      thinking: false,
      shift: 3,
      infer_method: "ode",
      use_adg: false,
      seed: -1,
      hm_temperature: 1.0,
      hm_cfg_scale: 1.5,
      hm_topk: 50,
    },
  });

  const selectedEngine = useWatch({ control: form.control, name: "engine" });
  const isHeartMuLa = selectedEngine === "heartmula";

  const mutation = useMutation({
    mutationFn: async (data: PlaygroundFormValues) => {
      const body: Record<string, any> = {
        engine: data.engine,
        prompt: data.prompt.trim(),
        duration: data.duration,
      };
      if (data.lyrics?.trim()) body.lyrics = data.lyrics.trim();
      if (data.style) body.style = data.style;
      if (data.seed !== -1) body.seed = data.seed;

      if (isHeartMuLa) {
        if (data.tags?.trim()) body.tags = data.tags.trim();
        if (data.negative_tags?.trim()) body.negative_tags = data.negative_tags.trim();
        if (data.title?.trim()) body.title = data.title.trim();
        body.temperature = data.hm_temperature;
        body.cfg_scale = data.hm_cfg_scale;
        body.topk = data.hm_topk;
      } else {
        if (data.instrument) body.instrument = data.instrument;
        if (data.bpm) body.bpm = data.bpm;
        if (data.key_scale) body.key_scale = data.key_scale;
        if (data.time_signature) body.time_signature = data.time_signature;
        if (data.vocal_language) body.vocal_language = data.vocal_language;
        body.inference_steps = data.inference_steps;
        body.guidance_scale = data.guidance_scale;
        if (data.thinking) body.thinking = true;
        if (data.shift !== 3) body.shift = data.shift;
        if (data.infer_method !== "ode") body.infer_method = data.infer_method;
        if (data.use_adg) body.use_adg = true;
      }

      const res = await apiRequest("POST", "/api/generate", body);
      return await res.json();
    },
    onSuccess: (data) => {
      setResponse(JSON.stringify(data, null, 2));
      queryClient.invalidateQueries({ queryKey: ["/api/jobs"] });
      toast({ title: "Job submitted", description: `Job ${data.id || "created"} successfully.` });
    },
    onError: (err: Error) => {
      setResponse(JSON.stringify({ error: err.message }, null, 2));
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  const handleSubmit = (values: PlaygroundFormValues) => {
    setResponse(null);
    mutation.mutate(values);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Form {...form}>
        <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-5">
          <FormField
            control={form.control}
            name="engine"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Engine</FormLabel>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => field.onChange("ace-step")}
                    className={`flex items-center gap-2 p-3 rounded-md border text-left transition-colors ${
                      field.value === "ace-step"
                        ? "border-blue-500 bg-blue-500/10"
                        : "border-border hover-elevate"
                    }`}
                    data-testid="button-engine-ace-step"
                  >
                    <Guitar className={`w-5 h-5 ${field.value === "ace-step" ? "text-blue-500" : "text-muted-foreground"}`} />
                    <div>
                      <div className="text-sm font-medium">ACE-Step</div>
                      <div className="text-xs text-muted-foreground">Instrumental & vocals</div>
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={() => field.onChange("heartmula")}
                    className={`flex items-center gap-2 p-3 rounded-md border text-left transition-colors ${
                      field.value === "heartmula"
                        ? "border-pink-500 bg-pink-500/10"
                        : "border-border hover-elevate"
                    }`}
                    data-testid="button-engine-heartmula"
                  >
                    <Mic className={`w-5 h-5 ${field.value === "heartmula" ? "text-pink-500" : "text-muted-foreground"}`} />
                    <div>
                      <div className="text-sm font-medium">HeartMuLa</div>
                      <div className="text-xs text-muted-foreground">Full songs with vocals</div>
                    </div>
                  </button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="prompt"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Prompt *</FormLabel>
                <FormControl>
                  <Textarea
                    placeholder={isHeartMuLa
                      ? "Describe the song: mood, genre, vocal style, instrumentation..."
                      : "Describe the music: genre, instruments, mood, tempo feel..."
                    }
                    className="resize-none min-h-[100px]"
                    data-testid="input-prompt"
                    {...field}
                  />
                </FormControl>
                <FormDescription className="text-xs">
                  {isHeartMuLa
                    ? 'e.g. "A dreamy love ballad with emotional female vocals and piano"'
                    : 'e.g. "Upbeat indie pop with jangly guitars and energetic female vocals"'
                  }
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="lyrics"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Lyrics {isHeartMuLa ? "(recommended)" : "(optional)"}</FormLabel>
                <FormControl>
                  <Textarea
                    placeholder={"[Verse 1]\nYour lyrics here...\n\n[Chorus]\nChorus lyrics..."}
                    className="resize-none min-h-[80px] font-mono text-sm"
                    data-testid="input-lyrics"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {isHeartMuLa ? (
            <>
              <FormField
                control={form.control}
                name="title"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Track Title (optional)</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="My Song Title"
                        data-testid="input-title"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="tags"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Style Tags</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Pop, female vocals, emotional, piano, ballad"
                        data-testid="input-tags"
                        {...field}
                      />
                    </FormControl>
                    <FormDescription className="text-xs">
                      Comma-separated style tags for genre, mood, instruments, vocal type
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="negative_tags"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Negative Tags (optional)</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="drums, distortion, noise, harsh"
                        data-testid="input-negative-tags"
                        {...field}
                      />
                    </FormControl>
                    <FormDescription className="text-xs">
                      Styles/elements to exclude from the generation
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="style"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Style</FormLabel>
                    <Select value={field.value} onValueChange={field.onChange}>
                      <FormControl>
                        <SelectTrigger data-testid="select-style">
                          <SelectValue placeholder="Select style" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {STYLES.map((s) => (
                          <SelectItem key={s} value={s}>{s}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="instrument"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Instrument</FormLabel>
                    <Select value={field.value} onValueChange={field.onChange}>
                      <FormControl>
                        <SelectTrigger data-testid="select-instrument">
                          <SelectValue placeholder="Select instrument" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {INSTRUMENTS.map((i) => (
                          <SelectItem key={i} value={i}>{i}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          )}

          <FormField
            control={form.control}
            name="duration"
            render={({ field }) => (
              <FormItem>
                <div className="flex items-center justify-between">
                  <FormLabel>Duration</FormLabel>
                  <span className="text-sm text-muted-foreground font-mono">{field.value}s</span>
                </div>
                <FormControl>
                  <Slider
                    value={[field.value]}
                    onValueChange={([v]) => field.onChange(v)}
                    min={10}
                    max={isHeartMuLa ? 300 : 300}
                    step={5}
                    data-testid="slider-duration"
                  />
                </FormControl>
                <div className="flex justify-between">
                  <span className="text-xs text-muted-foreground">10s</span>
                  <span className="text-xs text-muted-foreground">{isHeartMuLa ? "300s (5 min)" : "300s"}</span>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />

          {!isHeartMuLa && (
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="bpm"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>BPM (optional)</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        placeholder="Auto"
                        data-testid="input-bpm"
                        value={field.value ?? ""}
                        onChange={(e) => field.onChange(e.target.value ? parseInt(e.target.value) : undefined)}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="vocal_language"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Vocal Language</FormLabel>
                    <Select value={field.value || ""} onValueChange={(v) => field.onChange(v || "")}>
                      <FormControl>
                        <SelectTrigger data-testid="select-vocal-language">
                          <SelectValue placeholder="Auto" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {VOCAL_LANGUAGES.map((l) => (
                          <SelectItem key={l.value} value={l.value}>{l.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
            <FormField
              control={form.control}
              name="seed"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Seed</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      placeholder="-1 for random"
                      data-testid="input-seed"
                      {...field}
                      onChange={(e) => field.onChange(parseInt(e.target.value) || -1)}
                    />
                  </FormControl>
                  <p className="text-xs text-muted-foreground">-1 for random</p>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>

          {isHeartMuLa && (
            <div>
              <button
                type="button"
                className="flex items-center gap-2 text-sm text-muted-foreground hover-elevate rounded-md px-2 py-1.5 w-full"
                onClick={() => setShowAdvancedHeartMuLa(!showAdvancedHeartMuLa)}
                data-testid="button-toggle-hm-advanced"
              >
                <Settings className="w-4 h-4" />
                <span>Quality Settings</span>
                {showAdvancedHeartMuLa ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
              </button>

              {showAdvancedHeartMuLa && (
                <div className="mt-3 space-y-4 border rounded-md p-4">
                  <FormField
                    control={form.control}
                    name="hm_temperature"
                    render={({ field }) => (
                      <FormItem>
                        <div className="flex items-center justify-between">
                          <FormLabel>Temperature</FormLabel>
                          <span className="text-sm text-muted-foreground font-mono">{field.value.toFixed(1)}</span>
                        </div>
                        <FormControl>
                          <Slider
                            value={[field.value]}
                            onValueChange={([v]) => field.onChange(v)}
                            min={0.1}
                            max={2.0}
                            step={0.1}
                            data-testid="slider-hm-temperature"
                          />
                        </FormControl>
                        <p className="text-xs text-muted-foreground">Controls creativity: lower = more predictable, higher = more varied (default: 1.0)</p>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="hm_cfg_scale"
                    render={({ field }) => (
                      <FormItem>
                        <div className="flex items-center justify-between">
                          <FormLabel>CFG Scale</FormLabel>
                          <span className="text-sm text-muted-foreground font-mono">{field.value.toFixed(1)}</span>
                        </div>
                        <FormControl>
                          <Slider
                            value={[field.value]}
                            onValueChange={([v]) => field.onChange(v)}
                            min={0}
                            max={10}
                            step={0.1}
                            data-testid="slider-hm-cfg-scale"
                          />
                        </FormControl>
                        <p className="text-xs text-muted-foreground">Prompt adherence: higher = follows prompt more closely (default: 1.5)</p>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="hm_topk"
                    render={({ field }) => (
                      <FormItem>
                        <div className="flex items-center justify-between">
                          <FormLabel>Top-K Sampling</FormLabel>
                          <span className="text-sm text-muted-foreground font-mono">{field.value}</span>
                        </div>
                        <FormControl>
                          <Slider
                            value={[field.value]}
                            onValueChange={([v]) => field.onChange(v)}
                            min={1}
                            max={500}
                            step={1}
                            data-testid="slider-hm-topk"
                          />
                        </FormControl>
                        <p className="text-xs text-muted-foreground">Limits token choices: lower = more focused, higher = more diverse (default: 50)</p>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              )}
            </div>
          )}

          {!isHeartMuLa && (
            <div>
              <button
                type="button"
                className="flex items-center gap-2 text-sm text-muted-foreground hover-elevate rounded-md px-2 py-1.5 w-full"
                onClick={() => setShowAdvancedAceStep(!showAdvancedAceStep)}
                data-testid="button-toggle-advanced"
              >
                <Settings className="w-4 h-4" />
                <span>Advanced Settings</span>
                {showAdvancedAceStep ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
              </button>

              {showAdvancedAceStep && (
                <div className="mt-3 space-y-4 border rounded-md p-4">
                  <FormField
                    control={form.control}
                    name="inference_steps"
                    render={({ field }) => (
                      <FormItem>
                        <div className="flex items-center justify-between">
                          <FormLabel>Inference Steps</FormLabel>
                          <span className="text-sm text-muted-foreground font-mono">{field.value}</span>
                        </div>
                        <FormControl>
                          <Slider
                            value={[field.value]}
                            onValueChange={([v]) => field.onChange(v)}
                            min={1}
                            max={100}
                            step={1}
                            data-testid="slider-inference-steps"
                          />
                        </FormControl>
                        <p className="text-xs text-muted-foreground">Base model: 32-64 recommended. Turbo: 8</p>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="guidance_scale"
                    render={({ field }) => (
                      <FormItem>
                        <div className="flex items-center justify-between">
                          <FormLabel>Guidance Scale</FormLabel>
                          <span className="text-sm text-muted-foreground font-mono">{field.value}</span>
                        </div>
                        <FormControl>
                          <Slider
                            value={[field.value]}
                            onValueChange={([v]) => field.onChange(v)}
                            min={0}
                            max={30}
                            step={0.5}
                            data-testid="slider-guidance-scale"
                          />
                        </FormControl>
                        <p className="text-xs text-muted-foreground">Prompt adherence (default: 7). Base model only</p>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="key_scale"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Key / Scale</FormLabel>
                          <Select value={field.value || ""} onValueChange={(v) => field.onChange(v || "")}>
                            <FormControl>
                              <SelectTrigger data-testid="select-key-scale">
                                <SelectValue placeholder="Auto" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              {KEY_SCALES.map((k) => (
                                <SelectItem key={k} value={k}>{k}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="time_signature"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Time Signature</FormLabel>
                          <Select value={field.value || ""} onValueChange={(v) => field.onChange(v || "")}>
                            <FormControl>
                              <SelectTrigger data-testid="select-time-signature">
                                <SelectValue placeholder="Auto" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              {TIME_SIGNATURES.map((t) => (
                                <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>

                  <FormField
                    control={form.control}
                    name="shift"
                    render={({ field }) => (
                      <FormItem>
                        <div className="flex items-center justify-between">
                          <FormLabel>Shift</FormLabel>
                          <span className="text-sm text-muted-foreground font-mono">{field.value}</span>
                        </div>
                        <FormControl>
                          <Slider
                            value={[field.value]}
                            onValueChange={([v]) => field.onChange(v)}
                            min={1}
                            max={5}
                            step={0.1}
                            data-testid="slider-shift"
                          />
                        </FormControl>
                        <p className="text-xs text-muted-foreground">Timestep shift (default: 3). Base model only</p>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="infer_method"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Inference Method</FormLabel>
                          <Select value={field.value} onValueChange={field.onChange}>
                            <FormControl>
                              <SelectTrigger data-testid="select-infer-method">
                                <SelectValue />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="ode">ODE (Euler, faster)</SelectItem>
                              <SelectItem value="sde">SDE (Stochastic)</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>

                  <div className="space-y-3 pt-1">
                    <FormField
                      control={form.control}
                      name="thinking"
                      render={({ field }) => (
                        <FormItem className="flex items-center justify-between gap-4">
                          <div>
                            <FormLabel>Thinking Mode</FormLabel>
                            <p className="text-xs text-muted-foreground">Use LM to generate audio codes for enhanced quality</p>
                          </div>
                          <FormControl>
                            <Switch
                              checked={field.value}
                              onCheckedChange={field.onChange}
                              data-testid="switch-thinking"
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="use_adg"
                      render={({ field }) => (
                        <FormItem className="flex items-center justify-between gap-4">
                          <div>
                            <FormLabel>Adaptive Dual Guidance</FormLabel>
                            <p className="text-xs text-muted-foreground">Enhanced guidance for base model</p>
                          </div>
                          <FormControl>
                            <Switch
                              checked={field.value}
                              onCheckedChange={field.onChange}
                              data-testid="switch-use-adg"
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          <Button
            type="submit"
            className="w-full"
            disabled={mutation.isPending}
            data-testid="button-generate"
          >
            {mutation.isPending ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Send className="w-4 h-4 mr-2" />
            )}
            {mutation.isPending ? "Submitting..." : "Generate Music"}
          </Button>
        </form>
      </Form>

      <div>
        <div className="flex items-center gap-2 mb-3">
          <Terminal className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium">Response</span>
        </div>
        {mutation.isPending ? (
          <div className="bg-muted rounded-md p-8 flex flex-col items-center justify-center min-h-[200px]">
            <Loader2 className="w-8 h-8 text-muted-foreground animate-spin mb-3" />
            <p className="text-sm text-muted-foreground">Sending request...</p>
          </div>
        ) : response ? (
          <CodeBlock code={response} />
        ) : (
          <div className="bg-muted rounded-md p-8 flex flex-col items-center justify-center text-center min-h-[200px]">
            <FileJson className="w-10 h-10 text-muted-foreground/40 mb-3" />
            <p className="text-sm text-muted-foreground">
              Submit a request to see the API response here
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function useEstimatedProgress(job: Job): number | null {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (job.status !== "IN_PROGRESS") return;
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [job.status]);

  if (job.status !== "IN_PROGRESS") return null;
  if (job.progress != null && job.progress > 0) return job.progress;

  const createdMs = new Date(job.createdAt).getTime();
  const elapsedSec = (now - createdMs) / 1000;

  const estimatedTotalSec = job.engine === "heartmula"
    ? Math.max(job.duration * 8, 120)
    : Math.max(job.duration * 3, 60);

  const ratio = elapsedSec / estimatedTotalSec;
  const progress = Math.round(95 * (1 - Math.exp(-2.5 * ratio)));

  return Math.max(1, Math.min(progress, 95));
}

function JobRow({ job }: { job: Job }) {
  const [playing, setPlaying] = useState(false);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);
  const estimatedProgress = useEstimatedProgress(job);

  const togglePlay = () => {
    if (!job.outputUrl) return;
    if (playing && audioEl) {
      audioEl.pause();
      setPlaying(false);
    } else {
      const audio = new Audio(`/api/jobs/${job.id}/audio`);
      audio.onended = () => setPlaying(false);
      audio.onerror = () => setPlaying(false);
      audio.play();
      setAudioEl(audio);
      setPlaying(true);
    }
  };

  return (
    <div className="flex items-center gap-3 p-3 rounded-md hover-elevate" data-testid={`row-job-${job.id}`}>
      <div className="shrink-0">
        {job.status === "COMPLETED" && job.outputUrl ? (
          <Button size="icon" variant="ghost" onClick={togglePlay} data-testid={`button-play-${job.id}`}>
            {playing ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>
        ) : (
          <div className="w-9 h-9 flex items-center justify-center">
            <Music className="w-4 h-4 text-muted-foreground/40" />
          </div>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate" data-testid={`text-job-prompt-${job.id}`}>{job.prompt}</p>
        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
          <EngineBadge engine={job.engine} />
          <span className="text-xs text-muted-foreground font-mono">
            {new Date(job.createdAt).toLocaleString()}
          </span>
          {job.style && (
            <Badge variant="outline" className="text-xs">{job.style}</Badge>
          )}
          <span className="text-xs text-muted-foreground">{job.duration}s</span>
        </div>
        {job.status === "IN_PROGRESS" && estimatedProgress != null && (
          <div className="mt-1.5 flex items-center gap-2" data-testid={`progress-bar-${job.id}`}>
            <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{
                  width: `${estimatedProgress}%`,
                  background: job.engine === "heartmula"
                    ? "linear-gradient(90deg, #ec4899, #f472b6)"
                    : "linear-gradient(90deg, #3b82f6, #60a5fa)",
                }}
              />
            </div>
            <span className="text-xs text-muted-foreground tabular-nums shrink-0" data-testid={`text-progress-${job.id}`}>~{estimatedProgress}%</span>
          </div>
        )}
        {job.status === "FAILED" && job.errorMessage && (
          <p className="text-xs text-destructive mt-1 truncate">{job.errorMessage}</p>
        )}
      </div>
      <div className="shrink-0 flex items-center gap-2">
        <StatusBadge status={job.status} />
        {job.status === "COMPLETED" && job.outputUrl && (
          <a href={`/api/jobs/${job.id}/audio`} download data-testid={`link-download-${job.id}`}>
            <Button size="icon" variant="ghost">
              <Download className="w-4 h-4" />
            </Button>
          </a>
        )}
      </div>
    </div>
  );
}

function HistoryTab() {
  const { data: jobs, isLoading, isError, error, refetch } = useQuery<Job[]>({
    queryKey: ["/api/jobs"],
    refetchInterval: 5000,
  });

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex items-center gap-3 p-3">
            <Skeleton className="w-9 h-9 rounded-md" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
            <Skeleton className="h-6 w-20 rounded-md" />
          </div>
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <AlertCircle className="w-12 h-12 text-destructive/50 mb-4" />
        <h3 className="text-lg font-medium mb-1">Failed to load jobs</h3>
        <p className="text-sm text-muted-foreground max-w-sm mb-4">
          {(error as Error)?.message || "An error occurred while fetching jobs."}
        </p>
        <Button variant="outline" onClick={() => refetch()} data-testid="button-retry-jobs">
          <RefreshCw className="w-4 h-4 mr-1" />
          Retry
        </Button>
      </div>
    );
  }

  if (!jobs || jobs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <History className="w-12 h-12 text-muted-foreground/30 mb-4" />
        <h3 className="text-lg font-medium mb-1">No jobs yet</h3>
        <p className="text-sm text-muted-foreground max-w-sm">
          Submit a music generation request from the Playground to see your jobs here.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <p className="text-sm text-muted-foreground">{jobs.length} job{jobs.length !== 1 ? "s" : ""}</p>
        <Button variant="ghost" size="sm" onClick={() => refetch()} data-testid="button-refresh-jobs">
          <RefreshCw className="w-4 h-4 mr-1" />
          Refresh
        </Button>
      </div>
      <Card className="divide-y divide-border">
        {jobs.map((job) => (
          <JobRow key={job.id} job={job} />
        ))}
      </Card>
    </div>
  );
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function PodCard({ name, icon: Icon, color, data, gpuSettings }: {
  name: string;
  icon: typeof Guitar;
  color: string;
  data: any;
  gpuSettings?: any;
}) {
  const isOnline = data?.configured && (data?.health?.status === "ok" || data?.health?.data?.status === "ok");
  const gpu = data?.runpod?.runtime?.gpus?.[0];
  const machine = data?.runpod?.machine;
  const runtime = data?.runpod?.runtime;
  const stats = data?.stats?.data;

  return (
    <Card className="p-0 overflow-visible">
      <div className="p-4 pb-3">
        <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Icon className={`w-5 h-5 ${color}`} />
            <span className="font-semibold text-sm">{name}</span>
          </div>
          <Badge variant={isOnline ? "outline" : "destructive"} data-testid={`badge-pod-status-${name.toLowerCase().replace(/\s/g, "-")}`}>
            {isOnline ? (
              <><CheckCircle2 className="w-3 h-3 mr-1" /> Online</>
            ) : (
              <><AlertCircle className="w-3 h-3 mr-1" /> Offline</>
            )}
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
          <div>
            <span className="text-muted-foreground">Pod ID</span>
            <p className="font-mono truncate" data-testid={`text-pod-id-${name.toLowerCase().replace(/\s/g, "-")}`}>{data?.podId || "—"}</p>
          </div>
          <div>
            <span className="text-muted-foreground">GPU</span>
            <p className="font-medium" data-testid={`text-gpu-${name.toLowerCase().replace(/\s/g, "-")}`}>{machine?.gpuDisplayName || "—"}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Uptime</span>
            <p>{runtime?.uptimeInSeconds ? formatUptime(runtime.uptimeInSeconds) : "—"}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Status</span>
            <p>{data?.runpod?.desiredStatus || "—"}</p>
          </div>
          {gpu && (
            <>
              <div>
                <span className="text-muted-foreground">GPU Util</span>
                <p data-testid={`text-gpu-util-${name.toLowerCase().replace(/\s/g, "-")}`}>{gpu.gpuUtilPercent}%</p>
              </div>
              <div>
                <span className="text-muted-foreground">VRAM Util</span>
                <p data-testid={`text-vram-util-${name.toLowerCase().replace(/\s/g, "-")}`}>{gpu.memoryUtilPercent}%</p>
              </div>
            </>
          )}
          {stats && (
            <>
              <div>
                <span className="text-muted-foreground">Total Jobs</span>
                <p>{stats.jobs?.total ?? "—"}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Avg Time</span>
                <p>{stats.avg_job_seconds ? `${stats.avg_job_seconds.toFixed(1)}s` : "—"}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Succeeded</span>
                <p className="text-emerald-600 dark:text-emerald-400">{stats.jobs?.succeeded ?? "—"}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Failed</span>
                <p className="text-destructive">{stats.jobs?.failed ?? "—"}</p>
              </div>
            </>
          )}
        </div>
      </div>

      {gpuSettings && (
        <div className="border-t px-4 py-3">
          <p className="text-xs font-medium mb-2 flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5" /> GPU Settings
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">4-bit Quant</span>
              <Badge variant={gpuSettings.quantization_4bit === "false" ? "outline" : "destructive"} className="text-[10px] px-1.5 py-0">
                {gpuSettings.quantization_4bit === "false" ? "OFF" : "ON"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Seq Offload</span>
              <Badge variant={gpuSettings.sequential_offload === "false" ? "outline" : "destructive"} className="text-[10px] px-1.5 py-0">
                {gpuSettings.sequential_offload === "false" ? "OFF" : "ON"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Torch Compile</span>
              <Badge variant={gpuSettings.torch_compile === false ? "outline" : "destructive"} className="text-[10px] px-1.5 py-0">
                {gpuSettings.torch_compile === false ? "OFF" : "ON"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Model Loaded</span>
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                {data?.health?.model_loaded ? "YES" : "—"}
              </Badge>
            </div>
          </div>
        </div>
      )}

      <div className="border-t px-4 py-2">
        <p className="font-mono text-[10px] text-muted-foreground truncate" data-testid={`text-pod-url-${name.toLowerCase().replace(/\s/g, "-")}`}>
          {data?.baseUrl || "Not configured"}
        </p>
      </div>
    </Card>
  );
}

function AdminTab() {
  const { toast } = useToast();

  const { data: diagnostics, isLoading: diagLoading, refetch: refetchDiag } = useQuery<any>({
    queryKey: ["/api/pod/diagnostics"],
    refetchInterval: 15000,
  });

  const { data: gpuSettings, isLoading: gpuLoading, refetch: refetchGpu } = useQuery<any>({
    queryKey: ["/api/heartmula/gpu-settings"],
    refetchInterval: 15000,
  });

  const { data: health, refetch: refetchHealth } = useQuery<any>({
    queryKey: ["/api/health"],
    refetchInterval: 10000,
  });

  const reloadMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/heartmula/reload-gpu"),
    onSuccess: async (res) => {
      const data = await res.json();
      toast({ title: data.success ? "GPU model reloaded" : "Reload failed", description: data.message, variant: data.success ? "default" : "destructive" });
      refetchDiag();
      refetchGpu();
    },
    onError: (err: any) => {
      toast({ title: "Reload failed", description: err.message, variant: "destructive" });
    },
  });

  const scheduleReloadMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/heartmula/schedule-reload"),
    onSuccess: async (res) => {
      const data = await res.json();
      toast({ title: "Reload scheduled", description: data.message });
    },
    onError: (err: any) => {
      toast({ title: "Schedule failed", description: err.message, variant: "destructive" });
    },
  });

  const enforceGpuMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/heartmula/enforce-gpu"),
    onSuccess: async (res) => {
      const data = await res.json();
      toast({
        title: data.changed ? "GPU settings updated" : "Settings OK",
        description: data.changed ? `Changed: ${JSON.stringify(data.changes)}${data.reloadScheduled ? " — reload scheduled" : ""}` : "All GPU settings are correct",
      });
      refetchGpu();
    },
    onError: (err: any) => {
      toast({ title: "Enforce failed", description: err.message, variant: "destructive" });
    },
  });

  const refreshAll = () => {
    refetchDiag();
    refetchGpu();
    refetchHealth();
  };

  if (diagLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Server className="w-5 h-5" /> Pod Statistics
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">Live diagnostics from RunPod infrastructure</p>
        </div>
        <Button variant="outline" size="sm" onClick={refreshAll} data-testid="button-refresh-admin">
          <RefreshCw className="w-4 h-4 mr-1" /> Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {[
          { label: "API", ok: health?.api, icon: Activity },
          { label: "ACE-Step", ok: health?.aceStep, icon: Guitar },
          { label: "HeartMuLa", ok: health?.heartmula, icon: Mic },
        ].map((s) => (
          <Card key={s.label} className="p-3 flex items-center gap-3">
            <s.icon className={`w-4 h-4 ${s.ok ? "text-emerald-500" : "text-destructive"}`} />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium">{s.label}</p>
            </div>
            <Badge variant={s.ok ? "outline" : "destructive"} data-testid={`badge-health-${s.label.toLowerCase().replace(/\s/g, "-")}`}>
              {s.ok ? "Healthy" : "Down"}
            </Badge>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PodCard
          name="ACE-Step"
          icon={Guitar}
          color="text-blue-500 dark:text-blue-400"
          data={diagnostics?.aceStep}
        />
        <PodCard
          name="HeartMuLa"
          icon={Mic}
          color="text-pink-500 dark:text-pink-400"
          data={diagnostics?.heartmula}
          gpuSettings={gpuSettings}
        />
      </div>

      <div>
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4" /> HeartMuLa GPU Controls
        </h3>
        <Card className="p-4">
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => reloadMutation.mutate()}
              disabled={reloadMutation.isPending}
              data-testid="button-reload-gpu"
            >
              {reloadMutation.isPending ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <RotateCcw className="w-4 h-4 mr-1" />}
              Reload Model
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => scheduleReloadMutation.mutate()}
              disabled={scheduleReloadMutation.isPending}
              data-testid="button-schedule-reload"
            >
              {scheduleReloadMutation.isPending ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Clock className="w-4 h-4 mr-1" />}
              Schedule Reload
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => enforceGpuMutation.mutate()}
              disabled={enforceGpuMutation.isPending}
              data-testid="button-enforce-gpu"
            >
              {enforceGpuMutation.isPending ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Shield className="w-4 h-4 mr-1" />}
              Enforce GPU Settings
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Reload reloads model on GPU. Schedule queues a reload for when all jobs finish. Enforce checks and fixes GPU-only settings.
          </p>
        </Card>
      </div>

      {diagnostics && (
        <details className="group">
          <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground transition-colors flex items-center gap-1" data-testid="button-raw-diagnostics">
            <FileJson className="w-3.5 h-3.5" /> Raw Diagnostics JSON
          </summary>
          <div className="mt-2">
            <CodeBlock code={JSON.stringify(diagnostics, null, 2)} />
          </div>
        </details>
      )}
    </div>
  );
}

const compareFormSchema = z.object({
  engine: z.enum(["ace-step", "heartmula"]).default("heartmula"),
  prompt: z.string().min(1, "Prompt is required"),
  lyrics: z.string().optional().default(""),
  tags: z.string().optional().default(""),
  negative_tags: z.string().optional().default(""),
  title: z.string().optional().default(""),
  duration: z.number().min(10).max(300).default(30),
  sunoModel: z.enum(["V5", "V4_5ALL", "V4_5PLUS", "V4_5", "V4"]).default("V5"),
});

function CompareTab() {
  const { toast } = useToast();
  const [selectedComparison, setSelectedComparison] = useState<string | null>(null);

  const compareForm = useForm<z.infer<typeof compareFormSchema>>({
    resolver: zodResolver(compareFormSchema),
    defaultValues: {
      engine: "heartmula",
      prompt: "",
      lyrics: "",
      tags: "",
      negative_tags: "",
      title: "",
      duration: 30,
      sunoModel: "V5",
    },
  });

  const comparisons = useQuery<any[]>({
    queryKey: ["/api/comparisons"],
    refetchInterval: 5000,
  });

  const comparisonDetail = useQuery<any>({
    queryKey: ["/api/comparisons", selectedComparison],
    enabled: !!selectedComparison,
    refetchInterval: (query) => {
      const data = query.state.data as any;
      if (data && (data.status === "COMPLETED" || data.status === "FAILED")) return false;
      return 3000;
    },
  });

  const compareMutation = useMutation({
    mutationFn: async (data: any) => {
      const res = await apiRequest("POST", "/api/compare", data);
      return res.json();
    },
    onSuccess: (data: any) => {
      toast({ title: "Comparison started", description: `ID: ${data.id}` });
      setSelectedComparison(data.id);
      queryClient.invalidateQueries({ queryKey: ["/api/comparisons"] });
    },
    onError: (err: any) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  const generateIdeaMutation = useMutation({
    mutationFn: async (engine: string) => {
      const res = await apiRequest("POST", "/api/generate-song-idea", { engine });
      return res.json();
    },
    onSuccess: (idea: any) => {
      compareForm.setValue("prompt", idea.prompt);
      compareForm.setValue("lyrics", idea.lyrics);
      compareForm.setValue("tags", idea.tags);
      compareForm.setValue("title", idea.title);
      compareForm.setValue("negative_tags", idea.negative_tags || "");
      compareForm.setValue("duration", idea.duration || 60);
      toast({ title: "Song idea generated", description: `"${idea.title}" — ${idea.tags}` });
    },
    onError: (err: any) => {
      toast({ title: "Failed to generate idea", description: err.message, variant: "destructive" });
    },
  });

  const analysis = comparisonDetail.data?.geminiAnalysis as any;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-5">
          <div className="flex items-center justify-between gap-2 mb-4">
            <h3 className="text-sm font-semibold flex items-center gap-2" data-testid="text-compare-form-title">
              <Zap className="w-4 h-4" />
              New Comparison
            </h3>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => generateIdeaMutation.mutate(compareForm.getValues("engine"))}
              disabled={generateIdeaMutation.isPending}
              data-testid="button-generate-idea"
            >
              {generateIdeaMutation.isPending ? (
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              ) : (
                <Wand2 className="w-3.5 h-3.5 mr-1.5" />
              )}
              {generateIdeaMutation.isPending ? "Generating..." : "Gemini: Generate Idea"}
            </Button>
          </div>
          <form
            onSubmit={compareForm.handleSubmit((data) => compareMutation.mutate(data))}
            className="space-y-4"
          >
            <div>
              <label className="text-xs font-medium mb-1 block">Engine</label>
              <Select
                value={compareForm.watch("engine")}
                onValueChange={(v) => compareForm.setValue("engine", v as any)}
              >
                <SelectTrigger data-testid="select-compare-engine">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="heartmula">HeartMuLa</SelectItem>
                  <SelectItem value="ace-step">ACE-Step</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-xs font-medium mb-1 block">Suno Model</label>
              <Select
                value={compareForm.watch("sunoModel")}
                onValueChange={(v) => compareForm.setValue("sunoModel", v as any)}
              >
                <SelectTrigger data-testid="select-compare-suno-model">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="V5">Suno V5 (Latest)</SelectItem>
                  <SelectItem value="V4_5ALL">Suno V4.5 All</SelectItem>
                  <SelectItem value="V4_5PLUS">Suno V4.5+</SelectItem>
                  <SelectItem value="V4_5">Suno V4.5</SelectItem>
                  <SelectItem value="V4">Suno V4</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-xs font-medium mb-1 block">Prompt *</label>
              <Textarea
                {...compareForm.register("prompt")}
                placeholder="Describe the music you want to generate..."
                className="resize-none text-sm"
                rows={2}
                data-testid="textarea-compare-prompt"
              />
              {compareForm.formState.errors.prompt && (
                <p className="text-xs text-destructive mt-1">{compareForm.formState.errors.prompt.message}</p>
              )}
            </div>

            <div>
              <label className="text-xs font-medium mb-1 block">Tags / Style</label>
              <Input
                {...compareForm.register("tags")}
                placeholder="e.g. pop, upbeat, female vocal"
                className="text-sm"
                data-testid="input-compare-tags"
              />
            </div>

            <div>
              <label className="text-xs font-medium mb-1 block">Lyrics</label>
              <Textarea
                {...compareForm.register("lyrics")}
                placeholder="Optional lyrics..."
                className="resize-none text-sm"
                rows={3}
                data-testid="textarea-compare-lyrics"
              />
            </div>

            <div>
              <label className="text-xs font-medium mb-1 block">Title</label>
              <Input
                {...compareForm.register("title")}
                placeholder="Song title"
                className="text-sm"
                data-testid="input-compare-title"
              />
            </div>

            <div>
              <label className="text-xs font-medium mb-1 block">Duration: {compareForm.watch("duration")}s</label>
              <Slider
                value={[compareForm.watch("duration")]}
                onValueChange={([v]) => compareForm.setValue("duration", v)}
                min={10}
                max={300}
                step={5}
                data-testid="slider-compare-duration"
              />
            </div>

            <Button
              type="submit"
              className="w-full"
              disabled={compareMutation.isPending || !compareForm.watch("prompt")}
              data-testid="button-start-comparison"
            >
              {compareMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Zap className="w-4 h-4 mr-2" />
              )}
              Start A/B Comparison
            </Button>
          </form>
        </Card>

        <div className="space-y-4">
          <Card className="p-5">
            <h3 className="text-sm font-semibold mb-3" data-testid="text-comparison-history-title">
              Comparison History
            </h3>
            {comparisons.isLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
              </div>
            ) : !comparisons.data?.length ? (
              <p className="text-xs text-muted-foreground">No comparisons yet. Start one above.</p>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {comparisons.data.map((c: any) => (
                  <button
                    key={c.id}
                    onClick={() => setSelectedComparison(c.id)}
                    className={`w-full text-left p-3 rounded-md border transition-colors ${
                      selectedComparison === c.id
                        ? "border-primary bg-primary/5"
                        : "hover-elevate"
                    }`}
                    data-testid={`button-comparison-${c.id}`}
                  >
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      <div className="flex items-center gap-2 min-w-0">
                        <EngineBadge engine={c.engine} />
                        <span className="text-xs truncate">{c.prompt.substring(0, 40)}</span>
                      </div>
                      <StatusBadge status={c.status} />
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {new Date(c.createdAt).toLocaleString()}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>

      {selectedComparison && comparisonDetail.data && (
        <ComparisonResult comparison={comparisonDetail.data} />
      )}
    </div>
  );
}

function ScoreBar({ label, ours, suno }: { label: string; ours: number; suno: number }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium min-w-0 truncate">{label}</span>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-blue-600 dark:text-blue-400 font-mono">{ours}/10</span>
          <span className="text-muted-foreground">vs</span>
          <span className="text-orange-600 dark:text-orange-400 font-mono">{suno}/10</span>
        </div>
      </div>
      <div className="flex items-center gap-1 h-4">
        <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all"
            style={{ width: `${ours * 10}%` }}
          />
        </div>
        <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-orange-500 rounded-full transition-all"
            style={{ width: `${suno * 10}%` }}
          />
        </div>
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Ours</span>
        <span>Suno</span>
      </div>
    </div>
  );
}

function ComparisonResult({ comparison }: { comparison: any }) {
  const analysis = comparison.geminiAnalysis as any;
  const isComplete = comparison.status === "COMPLETED";
  const isFailed = comparison.status === "FAILED";
  const isAnalyzing = comparison.status === "ANALYZING";
  const isInProgress = comparison.status === "IN_PROGRESS";

  return (
    <Card className="p-5">
      <div className="flex items-center justify-between gap-2 mb-4 flex-wrap">
        <h3 className="text-sm font-semibold flex items-center gap-2" data-testid="text-comparison-result-title">
          <Activity className="w-4 h-4" />
          Comparison Result
        </h3>
        <StatusBadge status={comparison.status} />
      </div>

      {(isInProgress || isAnalyzing) && (
        <div className="text-center py-8 space-y-3">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            {isAnalyzing
              ? "Both tracks generated. Gemini is analyzing..."
              : `Generating tracks... Our: ${comparison.ourStatus} | Suno: ${comparison.sunoStatus}`}
          </p>
        </div>
      )}

      {isFailed && (
        <div className="text-center py-8">
          <AlertCircle className="w-8 h-8 mx-auto text-destructive mb-2" />
          <p className="text-sm text-destructive">{comparison.errorMessage || "Comparison failed"}</p>
        </div>
      )}

      {isComplete && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <EngineBadge engine={comparison.engine} />
                <span className="text-xs text-muted-foreground">Our Engine</span>
                {analysis && (
                  <Badge variant="outline" className="ml-auto text-xs">
                    {analysis.overallScore.ours}/10
                  </Badge>
                )}
              </div>
              <audio
                controls
                className="w-full"
                src={`/api/comparisons/${comparison.id}/audio/ours`}
                data-testid="audio-ours"
              />
            </Card>

            <Card className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <Badge variant="outline" className="text-xs bg-orange-500/10 text-orange-700 dark:text-orange-400 border-orange-500/20">
                  <Music className="w-3 h-3 mr-1" />
                  Suno {comparison.sunoModel}
                </Badge>
                <span className="text-xs text-muted-foreground">Reference</span>
                {analysis && (
                  <Badge variant="outline" className="ml-auto text-xs">
                    {analysis.overallScore.suno}/10
                  </Badge>
                )}
              </div>
              <audio
                controls
                className="w-full"
                src={`/api/comparisons/${comparison.id}/audio/suno`}
                data-testid="audio-suno"
              />
            </Card>
          </div>

          {analysis && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-4">
                  <h4 className="text-xs font-semibold mb-3">Overall Scores</h4>
                  <div className="flex items-center justify-center gap-8">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-blue-600 dark:text-blue-400" data-testid="text-score-ours">
                        {analysis.overallScore.ours}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">Ours</div>
                    </div>
                    <div className="text-xl text-muted-foreground">vs</div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-orange-600 dark:text-orange-400" data-testid="text-score-suno">
                        {analysis.overallScore.suno}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">Suno</div>
                    </div>
                  </div>
                </Card>

                <Card className="p-4">
                  <h4 className="text-xs font-semibold mb-3">Category Breakdown</h4>
                  <div className="space-y-3">
                    {analysis.categories?.map((cat: any) => (
                      <ScoreBar key={cat.name} label={cat.name} ours={cat.oursScore} suno={cat.sunoScore} />
                    ))}
                  </div>
                </Card>
              </div>

              <Card className="p-4">
                <h4 className="text-xs font-semibold mb-2">Summary</h4>
                <p className="text-sm text-muted-foreground" data-testid="text-analysis-summary">{analysis.summary}</p>
              </Card>

              <Card className="p-4">
                <h4 className="text-xs font-semibold mb-2">Detailed Comparison</h4>
                <p className="text-sm text-muted-foreground" data-testid="text-analysis-detailed">{analysis.detailedComparison}</p>
              </Card>

              {analysis.recommendations?.length > 0 && (
                <Card className="p-4">
                  <h4 className="text-xs font-semibold mb-2">Recommendations to Improve</h4>
                  <ul className="space-y-1">
                    {analysis.recommendations.map((rec: string, i: number) => (
                      <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 text-green-500 shrink-0" />
                        <span data-testid={`text-recommendation-${i}`}>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}

              {analysis.categories?.length > 0 && (
                <Card className="p-4">
                  <h4 className="text-xs font-semibold mb-2">Per-Category Analysis</h4>
                  <div className="space-y-3">
                    {analysis.categories.map((cat: any) => (
                      <div key={cat.name} className="border-b last:border-0 pb-2 last:pb-0">
                        <div className="flex items-center justify-between gap-2 mb-1 flex-wrap">
                          <span className="text-xs font-medium">{cat.name}</span>
                          <div className="flex items-center gap-2 text-xs">
                            <span className="text-blue-600 dark:text-blue-400">{cat.oursScore}/10</span>
                            <span className="text-muted-foreground">vs</span>
                            <span className="text-orange-600 dark:text-orange-400">{cat.sunoScore}/10</span>
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground">{cat.analysis}</p>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      )}
    </Card>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b sticky top-0 z-50 bg-background/95 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center">
              <Music className="w-4 h-4 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-sm font-semibold leading-tight">Music Gen API</h1>
              <p className="text-xs text-muted-foreground leading-tight">ACE-Step + HeartMuLa</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="font-mono text-xs">multi-engine</Badge>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-2" data-testid="text-page-title">
            Music Generation API
          </h2>
          <p className="text-muted-foreground max-w-2xl">
            Generate music using ACE-Step v1.5 (instrumental & vocal tracks) or HeartMuLa (full songs with vocals up to 5 minutes).
            Both engines run on RunPod GPU infrastructure.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {[
            { icon: Guitar, title: "ACE-Step v1.5", desc: "Fine-grained DiT music generation", color: "text-blue-500 dark:text-blue-400" },
            { icon: Mic, title: "HeartMuLa", desc: "Full songs with vocals & lyrics", color: "text-pink-500 dark:text-pink-400" },
            { icon: Zap, title: "GPU-Accelerated", desc: "RunPod A100/RTX inference", color: "text-amber-500 dark:text-amber-400" },
          ].map((f) => (
            <Card key={f.title} className="p-4">
              <f.icon className={`w-5 h-5 ${f.color} mb-2`} />
              <h3 className="text-sm font-semibold" data-testid={`text-feature-${f.title.toLowerCase().replace(/[\s.]/g, "-")}`}>{f.title}</h3>
              <p className="text-xs text-muted-foreground mt-0.5">{f.desc}</p>
            </Card>
          ))}
        </div>

        <Tabs defaultValue="playground" className="w-full">
          <TabsList className="mb-6">
            <TabsTrigger value="playground" data-testid="tab-playground">
              <Terminal className="w-4 h-4 mr-1.5" />
              Playground
            </TabsTrigger>
            <TabsTrigger value="docs" data-testid="tab-docs">
              <BookOpen className="w-4 h-4 mr-1.5" />
              API Docs
            </TabsTrigger>
            <TabsTrigger value="history" data-testid="tab-history">
              <History className="w-4 h-4 mr-1.5" />
              Job History
            </TabsTrigger>
            <TabsTrigger value="compare" data-testid="tab-compare">
              <Zap className="w-4 h-4 mr-1.5" />
              Compare
            </TabsTrigger>
            <TabsTrigger value="admin" data-testid="tab-admin">
              <Server className="w-4 h-4 mr-1.5" />
              Admin
            </TabsTrigger>
          </TabsList>

          <TabsContent value="playground">
            <PlaygroundTab />
          </TabsContent>
          <TabsContent value="docs">
            <ApiDocsTab />
          </TabsContent>
          <TabsContent value="history">
            <HistoryTab />
          </TabsContent>
          <TabsContent value="compare">
            <CompareTab />
          </TabsContent>
          <TabsContent value="admin">
            <AdminTab />
          </TabsContent>
        </Tabs>
      </main>

      <footer className="border-t mt-12">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 flex items-center justify-between flex-wrap gap-2">
          <p className="text-xs text-muted-foreground">
            Powered by ACE-Step v1.5 + HeartMuLa on RunPod
          </p>
          <div className="flex items-center gap-3">
            <a
              href="https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              ACE-Step
            </a>
            <a
              href="https://github.com/HeartMuLa/heartlib"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              HeartMuLa
            </a>
            <a
              href="https://www.runpod.io/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              RunPod
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
