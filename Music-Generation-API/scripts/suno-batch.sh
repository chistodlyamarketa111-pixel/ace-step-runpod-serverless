#!/bin/bash
set -e

SUNO_API_KEY="${SUNO_API_KEY}"
BASE_URL="https://api.sunoapi.org"
CASES_FILE="data/comparison_cases.json"
AUDIO_DIR="public/audio/suno"
LOG_FILE="/tmp/suno_batch.log"

mkdir -p "$AUDIO_DIR"

log() {
  echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

generate() {
  local id="$1" title="$2" style="$3" lyrics="$4"
  
  local outfile="$AUDIO_DIR/${id}.mp3"
  if [ -f "$outfile" ] && [ $(stat -f%z "$outfile" 2>/dev/null || stat -c%s "$outfile" 2>/dev/null) -gt 10000 ]; then
    log "SKIP $id (exists)"
    return 0
  fi

  local body
  body=$(python3 -c "
import json, sys
print(json.dumps({
  'customMode': True,
  'instrumental': False,
  'model': 'V5',
  'prompt': sys.argv[1],
  'style': sys.argv[2],
  'title': sys.argv[3],
  'callBackUrl': 'https://httpbin.org/post'
}))
" "$lyrics" "$style" "$title")

  log "SUBMIT $id: $title"
  
  local resp
  resp=$(curl -s -X POST "$BASE_URL/api/v1/generate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $SUNO_API_KEY" \
    -d "$body" --max-time 30)

  local code=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code',''))" 2>/dev/null)
  if [ "$code" != "200" ]; then
    log "FAIL $id: API error: $resp"
    return 1
  fi

  local taskId=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['taskId'])" 2>/dev/null)
  log "$id: taskId=$taskId"

  local elapsed=0
  local maxwait=600
  while [ $elapsed -lt $maxwait ]; do
    sleep 10
    elapsed=$((elapsed + 10))
    
    local status_resp
    status_resp=$(curl -s "$BASE_URL/api/v1/generate/record-info?taskId=$taskId" \
      -H "Authorization: Bearer $SUNO_API_KEY" --max-time 15 2>/dev/null)
    
    local status=$(echo "$status_resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('data',{}).get('status',''))" 2>/dev/null)
    
    if [ "$status" = "SUCCESS" ] || [ "$status" = "FIRST_SUCCESS" ] || [ "$status" = "CALLBACK_EXCEPTION" ]; then
      local audio_url=$(echo "$status_resp" | python3 -c "
import sys,json
d=json.load(sys.stdin)
tracks=d.get('data',{}).get('response',{}).get('sunoData',[])
if tracks:
  print(tracks[0].get('audioUrl','') or tracks[0].get('streamAudioUrl',''))
" 2>/dev/null)

      if [ -n "$audio_url" ] && [ "$audio_url" != "None" ]; then
        log "$id: downloading $audio_url"
        curl -sL "$audio_url" -o "$outfile" --max-time 60
        local size=$(stat -c%s "$outfile" 2>/dev/null || stat -f%z "$outfile" 2>/dev/null)
        log "$id: DONE (${size}B)"
        return 0
      fi
    fi

    if [ "$status" = "CREATE_TASK_FAILED" ] || [ "$status" = "GENERATE_AUDIO_FAILED" ] || [ "$status" = "SENSITIVE_WORD_ERROR" ]; then
      log "$id: FAILED status=$status"
      return 1
    fi
  done
  
  log "$id: TIMEOUT"
  return 1
}

log "Starting Suno batch generation"

total=$(python3 -c "import json; cases=json.load(open('$CASES_FILE')); print(len(cases))")
log "Total cases: $total"

done_count=0
fail_count=0

python3 -c "
import json
cases = json.load(open('$CASES_FILE'))
for c in cases:
    lyrics = c['lyrics'].replace('\n', '\\\\n')
    print(f\"{c['id']}|{c['title']}|{c['style_prompt']}|{lyrics}\")
" | while IFS='|' read -r id title style lyrics; do
  lyrics=$(echo "$lyrics" | sed 's/\\n/\n/g')
  
  if generate "$id" "$title" "$style" "$lyrics"; then
    done_count=$((done_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi
  
  sleep 2
done

log "BATCH COMPLETE"
