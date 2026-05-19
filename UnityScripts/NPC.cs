using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class CompanionNPC : MonoBehaviour
{
    [Header("Optional Custom Clips")]
    public List<VoiceLine> levelEntryLines = new();
    public VoiceLine panicDetectedLine;
    public VoiceLine stillCalmingLine;
    public VoiceLine userRecoveredLine;
    public VoiceLine regressionLine;

    int currentLevel = -1;

    bool shouldFollow = false;
    bool positionLocked = false;
    Vector3 lockedPosition;

    Dictionary<NpcState, List<int>> usedIndices = new Dictionary<NpcState, List<int>>()
    {
        { NpcState.Calm, new List<int>() },
        { NpcState.Elevated, new List<int>() },
        { NpcState.Panic, new List<int>() }
    };

    Dictionary<NpcState, int> lastPlayedIndex = new Dictionary<NpcState, int>()
    {
        { NpcState.Calm, -1 },
        { NpcState.Elevated, -1 },
        { NpcState.Panic, -1 }
    };

    [Header("Voice")]
    public AudioSource audioSource;

    [System.Serializable]
    public class VoiceLine
    {
        public AudioClip clip;
    }

    [Header("References")]
    public Transform player;
    public BiofeedbackMQTT bio;
    public FakeBiofeedbackMQTT fakeBio;
    public BaselineManager baseline;

    [Header("Follow Settings")]
    public Vector3 offsetFromPlayer = new Vector3(0.8f, 0f, -0.8f);
    public float followSmooth = 8f;
    public float rotateSmooth = 10f;
    public bool facePlayerWhenTalking = true;

    [Header("Stress Thresholds")]
    public float elevatedPercent = 0.20f;
    public float panicPercent = 0.40f;
    public float calmHysteresisPercent = 0.12f;

    [Header("Talk Pacing")]
    public float calmInterval = 12f;
    public float elevatedInterval = 7f;
    public float panicInterval = 4f;

    [Header("GSR Influence")]
    public bool useGsrToIntensify = true;

    [Header("Audio Behaviour")]
    public bool interruptCurrentVoice = false;

    enum NpcState { Unknown, Calm, Elevated, Panic }
    NpcState currentState = NpcState.Unknown;

    float lastTalkTime = -999f;

    [Header("Voice Clips By State")]
    public List<VoiceLine> calmLines = new();
    public List<VoiceLine> elevatedLines = new();
    public List<VoiceLine> panicLines = new();
    public List<VoiceLine> avoidanceLines = new();

    public VoiceLine congratsLine;

    List<int> usedAvoidanceIndices = new List<int>();
    int lastAvoidanceIndex = -1;

    Queue<VoiceLine> essentialQueue = new Queue<VoiceLine>();

    // ─── Unity ───────────────────────────────────────────────────────────────

    void Start()
    {
        if (!bio) bio = FindFirstObjectByType<BiofeedbackMQTT>();
        if (!fakeBio) fakeBio = FindFirstObjectByType<FakeBiofeedbackMQTT>();
        if (!baseline) baseline = BaselineManager.Instance;
        if (!audioSource) audioSource = GetComponent<AudioSource>();

        if (!player) Debug.LogWarning("CompanionNPC: Player not assigned.");
        if (!audioSource) Debug.LogWarning("CompanionNPC: No AudioSource assigned.");
    }

    void LateUpdate()
    {
        if (currentLevel == 0)
        {
            FollowPlayer();
        }
        else
        {
            if (!positionLocked && shouldFollow)
                FollowPlayer();
            else if (positionLocked)
                transform.position = lockedPosition;
        }

        if (currentLevel == 0) return;

        DrainEssentialQueue();
        UpdateAndTalk();
    }

    // ─── Public Notify API ───────────────────────────────────────────────────

    public void NotifyLevelEntered(int level)
    {
        if (currentLevel == level) return;

        CancelInvoke(nameof(LockPosition));

        currentLevel = level;
        currentState = NpcState.Unknown;
        ResetUsedLines();

        if (level == 0)
        {
            shouldFollow = true;
            positionLocked = false;
        }
        else
        {
            shouldFollow = false;
            positionLocked = true;
            lockedPosition = transform.position;
        }

        if (levelEntryLines != null && level >= 0 && level < levelEntryLines.Count)
            PlayEssentialLine(levelEntryLines[level]);
        else
            Debug.LogWarning($"CompanionNPC: No entry clip assigned for level {level}.");
    }

    public void NotifyCongrats()
    {
        PlayEssentialLine(congratsLine, clearQueue: true);
    }

    public void NotifyRegression(int targetLevel)
    {
        PlayEssentialLine(regressionLine, clearQueue: true);
    }

    public void NotifyPanicDetected()
    {
        PlayEssentialLine(panicDetectedLine);
    }

    public void NotifyUserRecovered()
    {
        PlayEssentialLine(userRecoveredLine);
    }

    public void NotifyStillCalming(float secondsLeft)
    {
        PlayEssentialLine(stillCalmingLine);
    }

    public void NotifyAvoidingStimulus()
    {
        VoiceLine line = GetRandomAvoidanceLine();
        PlayEssentialLine(line);
    }

    public float GetRemainingAudioTime()
    {
        if (audioSource == null || !audioSource.isPlaying) return 0f;
        return audioSource.clip != null
            ? audioSource.clip.length - audioSource.time
            : 0f;
    }

    public void ClearQueue()
    {
        essentialQueue.Clear();
    }

    // ─── Position Helpers ────────────────────────────────────────────────────

    public void SetFixedPosition(Transform npcSpawn)
    {
        if (npcSpawn == null) return;

        CancelInvoke(nameof(LockPosition));

        transform.position = npcSpawn.position;
        transform.rotation = npcSpawn.rotation;

        lockedPosition = npcSpawn.position;
        positionLocked = true;
        shouldFollow = false;
    }

    public void EnableFollowMode()
    {
        CancelInvoke(nameof(LockPosition));
        positionLocked = false;
        shouldFollow = true;
    }

    void LockPosition()
    {
        lockedPosition = transform.position;
        positionLocked = true;
        shouldFollow = false;
    }

    void FollowPlayer()
    {
        if (!player) return;

        Vector3 flatForward = player.forward;
        flatForward.y = 0f;
        flatForward.Normalize();

        Vector3 flatRight = player.right;
        flatRight.y = 0f;
        flatRight.Normalize();

        Vector3 desiredPos =
            player.position +
            flatRight * offsetFromPlayer.x +
            Vector3.up * offsetFromPlayer.y +
            flatForward * offsetFromPlayer.z;

        transform.position = Vector3.Lerp(
            transform.position,
            desiredPos,
            1f - Mathf.Exp(-followSmooth * Time.deltaTime)
        );

        Vector3 targetDir;

        if (facePlayerWhenTalking && (Time.time - lastTalkTime) < 2.5f)
            targetDir = player.position - transform.position;
        else
            targetDir = flatForward;

        targetDir.y = 0f;

        if (targetDir.sqrMagnitude > 0.001f)
        {
            Quaternion targetRot = Quaternion.LookRotation(targetDir.normalized);
            transform.rotation = Quaternion.Slerp(
                transform.rotation,
                targetRot,
                1f - Mathf.Exp(-rotateSmooth * Time.deltaTime)
            );
        }
    }

    // ─── State & Talking ─────────────────────────────────────────────────────

    void UpdateAndTalk()
    {
        int bpm = GetBPM();
        if (bpm <= 0) return;

        float baseBpm = (baseline != null && baseline.baselineBPM > 0f)
            ? baseline.baselineBPM
            : 70f;

        float elevatedThresh = baseBpm * (1f + elevatedPercent);
        float panicThresh = baseBpm * (1f + panicPercent);
        float calmReturnThresh = baseBpm * (1f + calmHysteresisPercent);

        GSRStatee gsr = GetGSR();
        bool gsrStressed = (gsr == GSRStatee.Stressed);

        NpcState next = currentState;

        if (bpm >= panicThresh)
            next = NpcState.Panic;
        else if (bpm >= elevatedThresh)
            next = NpcState.Elevated;
        else if (bpm <= calmReturnThresh)
            next = NpcState.Calm;

        if (useGsrToIntensify && gsrStressed)
        {
            if (next == NpcState.Calm) next = NpcState.Elevated;
            else if (next == NpcState.Elevated) next = NpcState.Panic;
        }

        if (next == NpcState.Unknown)
            next = NpcState.Calm;

        if (next != currentState)
        {
            currentState = next;

            if (Time.time - lastTalkTime >= 2f)
                PlayLineForState(currentState);

            return;
        }

        float interval =
            currentState == NpcState.Panic ? panicInterval :
            currentState == NpcState.Elevated ? elevatedInterval :
            calmInterval;

        if (Time.time - lastTalkTime >= interval)
            PlayLineForState(currentState);
    }

    void PlayLineForState(NpcState state)
    {
        VoiceLine selected = state switch
        {
            NpcState.Calm => GetRandomLineNoRepeat(calmLines, state),
            NpcState.Elevated => GetRandomLineNoRepeat(elevatedLines, state),
            NpcState.Panic => GetRandomLineNoRepeat(panicLines, state),
            _ => null
        };

        if (selected == null || selected.clip == null) return;

        lastTalkTime = Time.time;

        if (interruptCurrentVoice)
        {
            audioSource.Stop();
            audioSource.clip = selected.clip;
            audioSource.Play();
        }
        else
        {
            if (!audioSource.isPlaying)
                audioSource.PlayOneShot(selected.clip);
        }
    }

    // ─── Audio Queue ─────────────────────────────────────────────────────────

    void PlayEssentialLine(VoiceLine line, bool clearQueue = false)
    {
        if (line == null || line.clip == null) return;

        if (clearQueue)
            essentialQueue.Clear();

        if (interruptCurrentVoice)
        {
            essentialQueue.Clear();
            audioSource.Stop();
            audioSource.clip = line.clip;
            audioSource.Play();
            lastTalkTime = Time.time;
        }
        else
        {
            if (!audioSource.isPlaying)
            {
                audioSource.PlayOneShot(line.clip);
                lastTalkTime = Time.time;
            }
            else
            {
                essentialQueue.Enqueue(line);
            }
        }
    }

    void DrainEssentialQueue()
    {
        if (audioSource.isPlaying) return;
        if (essentialQueue.Count == 0) return;

        VoiceLine next = essentialQueue.Dequeue();
        if (next?.clip == null) return;

        audioSource.PlayOneShot(next.clip);
        lastTalkTime = Time.time;
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    int GetBPM()
    {
        if (bio != null) return bio.currentBPM;
        if (fakeBio != null) return fakeBio.currentBPM;
        return 0;
    }

    GSRStatee GetGSR()
    {
        if (fakeBio != null) return fakeBio.GetGSRState();
        return GSRStatee.Calm;
    }

    VoiceLine GetRandomLineNoRepeat(List<VoiceLine> lines, NpcState state)
    {
        if (lines == null || lines.Count == 0) return null;

        var used = usedIndices[state];

        if (used.Count >= lines.Count)
            used.Clear();

        List<int> available = new List<int>();
        for (int i = 0; i < lines.Count; i++)
            if (!used.Contains(i))
                available.Add(i);

        int index;
        do { index = available[Random.Range(0, available.Count)]; }
        while (available.Count > 1 && index == lastPlayedIndex[state]);

        lastPlayedIndex[state] = index;
        used.Add(index);

        return lines[index];
    }

    VoiceLine GetRandomAvoidanceLine()
    {
        if (avoidanceLines == null || avoidanceLines.Count == 0) return null;

        if (usedAvoidanceIndices.Count >= avoidanceLines.Count)
            usedAvoidanceIndices.Clear();

        List<int> available = new List<int>();
        for (int i = 0; i < avoidanceLines.Count; i++)
            if (!usedAvoidanceIndices.Contains(i))
                available.Add(i);

        int index;
        do { index = available[Random.Range(0, available.Count)]; }
        while (available.Count > 1 && index == lastAvoidanceIndex);

        lastAvoidanceIndex = index;
        usedAvoidanceIndices.Add(index);

        return avoidanceLines[index];
    }

    void ResetUsedLines()
    {
        foreach (var key in usedIndices.Keys)
            usedIndices[key].Clear();
    }
}