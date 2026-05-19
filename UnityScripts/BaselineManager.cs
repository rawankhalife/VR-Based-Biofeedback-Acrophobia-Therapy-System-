using System.Collections.Generic;
using UnityEngine;

// -----------------------------------------------------------------------
// GSRState (Enum)
// Represents the user's Galvanic Skin Response stress state.
//   - RelaxedOrNormal : User is calm, no stress detected.
//   - Stressed        : User is exhibiting physiological stress.
//   - Invalid         : Reading is unavailable or could not be determined.
// -----------------------------------------------------------------------
public enum GSRState
{
    RelaxedOrNormal,
    Stressed,
    Invalid
}

// -----------------------------------------------------------------------
// BaselineManager
// Singleton MonoBehaviour that manages biofeedback baseline collection.
// Persists across scenes via DontDestroyOnLoad.
//
// Responsibilities:
//   - Collects BPM and GSR samples over a timed window.
//   - Computes and stores the player's baseline BPM and GSR state.
//   - Coordinates phase/state transitions with BiofeedbackMQTT.
//   - On subsequent runs, treats collection as a recovery relaxation phase
//     rather than a fresh baseline.
// -----------------------------------------------------------------------
public class BaselineManager : MonoBehaviour
{
    public static BaselineManager Instance;

    [Header("Baseline Results")]
    public float baselineBPM;
    public GSRState baselineGSRState;

    private List<int> bpmSamples = new List<int>();
    private List<GSRState> gsrSamples = new List<GSRState>();

    private bool collecting = false;
    private BiofeedbackMQTT bio;
    private bool baselineAlreadyCompleted = false;


    // -----------------------------------------------------------------------
    // Awake()
    // Enforces the singleton pattern.
    // Destroys any duplicate instance that gets created on scene reload.
    // Marks this object to persist across scene loads.
    // -----------------------------------------------------------------------
    void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    // -----------------------------------------------------------------------
    // Start()
    // Attempts to locate BiofeedbackMQTT in the scene at startup.
    // Logs a warning if not found since data collection depends on it.
    // -----------------------------------------------------------------------
    void Start()
    {
        bio = FindObjectOfType<BiofeedbackMQTT>();

        if (bio == null)
            Debug.LogWarning("BaselineManager could not find BiofeedbackMQTT in the scene!");
    }

    // -----------------------------------------------------------------------
    // Update()
    // Runs every frame. If collection is active, samples the current BPM
    // and GSR state from BiofeedbackMQTT each frame.
    // Also performs a lazy re-lookup of BiofeedbackMQTT if reference is missing.
    // -----------------------------------------------------------------------
    void Update()
    {
        if (bio == null)
            bio = FindObjectOfType<BiofeedbackMQTT>();

        if (!collecting || bio == null) return;

        bpmSamples.Add(bio.currentBPM);
        gsrSamples.Add(bio.GetGSRState());
    }

    // -----------------------------------------------------------------------
    // StartBaseline()
    // Begins a biofeedback data collection window.
    //
    // Behavior:
    //   - Clears any previously collected samples before starting.
    //   - First call  : enters 'baseline' phase, sets collectingBaseline = true.
    //   - Later calls : enters 'recovery_relaxation' phase instead.
    //   - Starts the MQTT session if it is not already active.
    //   - Has no effect if collection is already in progress.
    // -----------------------------------------------------------------------
    public void StartBaseline()
    {
        if (collecting)
        {
            Debug.Log("Baseline/recovery collection already active.");
            return;
        }

        if (bio == null)
            bio = FindObjectOfType<BiofeedbackMQTT>();

        bpmSamples.Clear();
        gsrSamples.Clear();
        collecting = true;

        if (bio != null)
        {
            if (!bio.IsSessionActive)
                bio.StartSession();

            if (!baselineAlreadyCompleted)
            {
                bio.collectingBaseline = true;
                bio.SetPhase("baseline");
                bio.SetLevel(0);
                bio.SetPath("None");
                bio.SetSystemState("Calm");
                bio.LogEvent("BaselineStarted", "Initial baseline collection started", 0);
            }
            else
            {
                bio.collectingBaseline = false;
                bio.SetPhase("recovery_relaxation");
                bio.SetSystemState("Calm");
                bio.LogEvent("RecoveryRelaxationStarted", "Returned to relaxation after panic", 0);
            }
        }

        Debug.Log("Baseline/recovery collection started");
    }

    // -----------------------------------------------------------------------
    // StopBaseline()
    // Ends the active collection window and processes the gathered samples.
    //
    // Behavior:
    //   - First call  : computes baselineBPM and baselineGSRState from samples,
    //                   marks baseline as completed, transitions to 'exposure' phase.
    //   - Later calls : ends recovery relaxation, transitions back to 'exposure'
    //                   without overwriting the stored baseline values.
    //   - If BiofeedbackMQTT is unavailable, computes baseline locally and continues.
    //   - Has no effect if collection is not currently active.
    // -----------------------------------------------------------------------
    public void StopBaseline()
    {
        if (!collecting)
        {
            Debug.Log("Baseline/recovery collection is not active.");
            return;
        }

        collecting = false;

        if (bio != null)
        {
            if (!baselineAlreadyCompleted)
            {
                bio.collectingBaseline = false;

                baselineBPM = ComputeAverage(bpmSamples);
                baselineGSRState = ComputeDominantGSRState();

                baselineAlreadyCompleted = true;

                bio.SetPhase("exposure");
                bio.SetLevel(0);
                bio.SetSystemState("Calm");
                bio.LogEvent("BaselineEnded", $"Baseline established | BPM: {baselineBPM}, GSR: {baselineGSRState}", 0);

                Debug.Log($"Baseline established | BPM: {baselineBPM}, GSR: {baselineGSRState}");
            }
            else
            {
                bio.collectingBaseline = false;
                bio.SetPhase("exposure");
                bio.SetSystemState("Calm");
                bio.LogEvent("RecoveryRelaxationEnded", "Recovery relaxation ended", 0);

                Debug.Log("Recovery relaxation ended");
            }
        }
        else
        {
            baselineBPM = ComputeAverage(bpmSamples);
            baselineGSRState = ComputeDominantGSRState();

            if (!baselineAlreadyCompleted)
                baselineAlreadyCompleted = true;

            Debug.Log($"Baseline established without BiofeedbackMQTT reference | BPM: {baselineBPM}, GSR: {baselineGSRState}");
        }
    }

    // -----------------------------------------------------------------------
    // ResetBaseline()
    // Fully resets all baseline data and internal state.
    //
    // Behavior:
    //   - Clears all collected BPM and GSR samples.
    //   - Resets baselineBPM to 0 and baselineGSRState to Invalid.
    //   - Resets baselineAlreadyCompleted so next collection is treated
    //     as a fresh baseline.
    //   - Notifies BiofeedbackMQTT to return to the 'idle' phase.
    // -----------------------------------------------------------------------
    public void ResetBaseline()
    {
        bpmSamples.Clear();
        gsrSamples.Clear();
        baselineBPM = 0f;
        baselineGSRState = GSRState.Invalid;
        baselineAlreadyCompleted = false;
        collecting = false;

        if (bio != null)
        {
            bio.collectingBaseline = false;
            bio.SetPhase("idle");
            bio.SetLevel(-1);
            bio.SetPath("None");
            bio.SetSystemState("Calm");
            bio.LogEvent("BaselineReset", "Baseline data reset");
        }

        Debug.Log("Baseline reset");
    }

    // -----------------------------------------------------------------------
    // ComputeAverage()
    // Calculates the arithmetic mean of a list of integer BPM samples.
    // Returns 0 if the list is empty to avoid division by zero.
    // -----------------------------------------------------------------------
    float ComputeAverage(List<int> values)
    {
        if (values.Count == 0) return 0f;

        float sum = 0f;
        foreach (int v in values)
            sum += v;

        return sum / values.Count;
    }

    // -----------------------------------------------------------------------
    // ComputeDominantGSRState()
    // Determines the dominant GSR state from collected samples using majority vote.
    // Counts RelaxedOrNormal vs Stressed samples, ignoring Invalid ones.
    // Returns RelaxedOrNormal if tied or relaxed samples are greater.
    // -----------------------------------------------------------------------
    GSRState ComputeDominantGSRState()
    {
        int relaxed = 0;
        int stressed = 0;

        foreach (var s in gsrSamples)
        {
            if (s == GSRState.RelaxedOrNormal)
                relaxed++;
            else if (s == GSRState.Stressed)
                stressed++;
        }

        return (relaxed >= stressed)
            ? GSRState.RelaxedOrNormal
            : GSRState.Stressed;
    }

    // -----------------------------------------------------------------------
    // BaselineAlreadyCompleted()
    // Returns true if the initial baseline has been established this session.
    // Used by other systems to check whether to treat collection as recovery.
    // -----------------------------------------------------------------------
    public bool BaselineAlreadyCompleted()
    {
        return baselineAlreadyCompleted;
    }
}
