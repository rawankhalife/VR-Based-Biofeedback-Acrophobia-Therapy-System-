using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class SceneProgressionManager : MonoBehaviour
{
    [Header("Relaxation Return")]
    [SerializeField, HideInInspector]
    private string relaxationSceneName;

    [Header("Player & Spawn Points")]
    public Transform player;
    public Transform level0Spawn;
    private bool isInSafeZone = true;

    [Header("Indoor Player Spawns")]
    public Transform indoorLevel1PlayerSpawn;
    public Transform indoorLevel2PlayerSpawn;
    public Transform indoorLevel3PlayerSpawn;

    [Header("Indoor NPC Spawns")]
    public Transform indoorLevel1NPCSpawn;
    public Transform indoorLevel2NPCSpawn;
    public Transform indoorLevel3NPCSpawn;

    [Header("Outdoor Player Spawns")]
    public Transform outdoorLevel1PlayerSpawn;
    public Transform outdoorLevel2PlayerSpawn;
    public Transform outdoorLevel3PlayerSpawn;

    [Header("Outdoor NPC Spawns")]
    public Transform outdoorLevel1NPCSpawn;
    public Transform outdoorLevel2NPCSpawn;
    public Transform outdoorLevel3NPCSpawn;

    [Header("UI Elements")]
    public GameObject proceedPrompt;
    public GameObject finalLevelPrompt;
    public GameObject pathChoicePrompt;

    [Header("Heart Rate Settings")]
    public float baselineBPM;
    public float maxIncreasePercent = 20f;
    public float panicSpikePercent = 40f;
    public float stableDuration = 30f;
    public float elevatedDuration = 15f;
    public float warningGracePeriod = 20f;
    public float panicSpikeWindow = 12f;

    [Header("Banner Positioning")]
    public Transform cameraTransform;
    public float bannerDistanceFromCamera = 0.8f;
    public float bannerHeightOffset = -0.2f;
    public bool faceCamera = true;

    private int currentLevel = 0;
    private bool waitingForUser = false;
    private bool panicHandling = false;
    private bool sessionComplete = false;
    private bool userWantsToProceed = false;

    private BiofeedbackMQTT bio;
    private CompanionNPC companion;
    private FakeBiofeedbackMQTT fakeBio;

    private Transform[] activePlayerSpawnPoints;
    private Transform[] activeNPCSpawnPoints;

    private GameObject currentFollowingBanner;

    public enum ExposurePath
    {
        None,
        Indoor,
        Outdoor
    }

    private ExposurePath chosenPath = ExposurePath.None;

    public void SetSafeZone(bool value)
    {
        isInSafeZone = value;

        if (bio != null)
        {
            bio.SetSafeZone(value);
            bio.LogEvent(value ? "SafeZoneEntered" : "SafeZoneExited", "", currentLevel);
        }
    }

    void Start()
    {
        if (proceedPrompt != null)
            proceedPrompt.SetActive(false);

        if (finalLevelPrompt != null)
            finalLevelPrompt.SetActive(false);

        if (pathChoicePrompt != null)
            pathChoicePrompt.SetActive(false);

        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;

        bio = FindFirstObjectByType<BiofeedbackMQTT>();
        fakeBio = FindFirstObjectByType<FakeBiofeedbackMQTT>();
        companion = FindFirstObjectByType<CompanionNPC>();

        baselineBPM = BaselineManager.Instance.baselineBPM;

        if (bio == null) Debug.LogWarning("No BiofeedbackMQTT found!");
        if (companion == null) Debug.LogWarning("No CompanionNPC found!");

        if (bio != null)
        {
            bio.collectingBaseline = false;
            bio.SetPhase("exposure");
            bio.SetLevel(0);
            bio.SetPath("None");
            bio.SetSafeZone(isInSafeZone);
            bio.SetSystemState("Calm");
            bio.LogEvent("Level0SceneStarted", "Entered Level 0 scene", 0);
        }

        TeleportPlayer(level0Spawn);
        companion?.NotifyLevelEntered(0);

        StartCoroutine(Level0Routine());

        Debug.Log("Bio found: " + (bio != null));
        Debug.Log("FakeBio found: " + (fakeBio != null));
    }

    void Update()
    {
        if (currentFollowingBanner != null && currentFollowingBanner.activeSelf)
        {
            UpdateBannerPosition(currentFollowingBanner);
        }
    }

    float GetBPM()
    {
        if (bio != null)
        {
            Debug.Log("Using BiofeedbackMQTT BPM: " + bio.currentBPM);
            return bio.currentBPM;
        }

        if (fakeBio != null)
        {
            Debug.Log("Using FakeBiofeedbackMQTT BPM: " + fakeBio.currentBPM);
            return fakeBio.currentBPM;
        }

        return 0;
    }

    IEnumerator Level0Routine()
    {
        currentLevel = 0;
        companion?.NotifyLevelEntered(0);

        bio?.SetPhase("exposure");
        bio?.SetLevel(0);
        bio?.SetSystemState("Calm");
        bio?.SetSafeZone(isInSafeZone);
        bio?.LogEvent("LevelEntered", "Entered Level 0", 0);

        yield return new WaitForSeconds(35f);

        waitingForUser = true;
        userWantsToProceed = false;
        chosenPath = ExposurePath.None;

        bio?.SetPhase("prompt");
        bio?.LogEvent("ProceedPromptShown", "Proceed prompt shown at Level 0", 0);

        ShowProceedPrompt();
        while (waitingForUser) yield return null;

        if (!userWantsToProceed)
        {
            StartCoroutine(Level0Routine());
            yield break;
        }

        if (chosenPath == ExposurePath.Indoor)
        {
            activePlayerSpawnPoints = new Transform[]
            {
                level0Spawn,
                indoorLevel1PlayerSpawn,
                indoorLevel2PlayerSpawn,
                indoorLevel3PlayerSpawn
            };

            activeNPCSpawnPoints = new Transform[]
            {
                null,
                indoorLevel1NPCSpawn,
                indoorLevel2NPCSpawn,
                indoorLevel3NPCSpawn
            };
        }
        else if (chosenPath == ExposurePath.Outdoor)
        {
            activePlayerSpawnPoints = new Transform[]
            {
                level0Spawn,
                outdoorLevel1PlayerSpawn,
                outdoorLevel2PlayerSpawn,
                outdoorLevel3PlayerSpawn
            };

            activeNPCSpawnPoints = new Transform[]
            {
                null,
                outdoorLevel1NPCSpawn,
                outdoorLevel2NPCSpawn,
                outdoorLevel3NPCSpawn
            };
        }
        else
        {
            StartCoroutine(Level0Routine());
            yield break;
        }

        bio?.SetPhase("exposure");
        bio?.SetLevel(1);
        bio?.LogEvent("LevelTransition", "Moved from Level 0 to Level 1", 1);

        MoveToLevel(1);
        StartCoroutine(LevelRoutine(1));
    }

    IEnumerator LevelRoutine(int level)
    {
        currentLevel = level;
        panicHandling = false;

        bio?.SetPhase("exposure");
        bio?.SetLevel(level);
        bio?.SetSafeZone(isInSafeZone);
        bio?.LogEvent("LevelEntered", $"Entered Level {level}", level);

        float stableTimer = 0f;
        float elevatedTimer = 0f;
        float spikeTimer = 0f;

        while (!sessionComplete)
        {
            if (panicHandling) yield break;

            float bpm = GetBPM();
            float elevatedThresh = baselineBPM * (1f + maxIncreasePercent / 100f);
            float spikeThresh = baselineBPM * (1f + panicSpikePercent / 100f);

            bool isElevated = bpm > elevatedThresh;
            bool isSpike = bpm > spikeThresh;

            if (bio != null)
            {
                if (isSpike)
                    bio.SetSystemState("Panic");
                else if (isElevated)
                    bio.SetSystemState("Elevated");
                else
                    bio.SetSystemState("Calm");
            }

            if (!isElevated && !isInSafeZone)
            {
                stableTimer += Time.deltaTime;
                elevatedTimer = 0f;
                spikeTimer = 0f;
            }
            else
            {
                stableTimer = 0f;
                elevatedTimer += isElevated ? Time.deltaTime : 0f;
                spikeTimer = isSpike ? spikeTimer + Time.deltaTime : 0f;
            }

            bool sustainedElevation = elevatedTimer >= elevatedDuration;
            bool truePanic = spikeTimer >= panicSpikeWindow;

            if (truePanic && !panicHandling)
            {
                panicHandling = true;

                bio?.LogEvent(
                    "TruePanicDetected",
                    $"Panic threshold exceeded for {panicSpikeWindow} seconds. BPM={bpm}",
                    level
                );
                bio?.SetSystemState("Panic");
                bio?.SetPhase("recovery_relaxation");

                StartCoroutine(ReturnToRelaxation());
                yield break;
            }

            if (sustainedElevation && !panicHandling)
            {
                panicHandling = true;

                bio?.LogEvent(
                    "SustainedElevationDetected",
                    $"Elevated threshold exceeded for {elevatedDuration} seconds. BPM={bpm}",
                    level
                );
                bio?.SetSystemState("Elevated");

                StartCoroutine(HandleElevation(level));
                yield break;
            }

            if (stableTimer >= stableDuration)
            {
                bio?.LogEvent(
                    "StableThresholdReached",
                    $"Stable for {stableDuration} seconds at Level {level}",
                    level
                );

                int nextLevel = level + 1;

                if (nextLevel < activePlayerSpawnPoints.Length)
                {
                    waitingForUser = true;
                    userWantsToProceed = false;

                    bio?.SetPhase("prompt");
                    bio?.LogEvent("ProceedPromptShown", $"Proceed prompt shown at Level {level}", level);

                    ShowProceedPrompt();
                    while (waitingForUser) yield return null;

                    if (userWantsToProceed)
                    {
                        bio?.LogEvent("ProceedAccepted", $"User accepted progression from Level {level} to Level {nextLevel}", level);
                        bio?.SetPhase("exposure");
                        bio?.SetLevel(nextLevel);

                        MoveToLevel(nextLevel);
                        StartCoroutine(LevelRoutine(nextLevel));
                    }
                    else
                    {
                        bio?.LogEvent("ProceedDeclined", $"User declined progression at Level {level}", level);
                        bio?.SetPhase("exposure");

                        StartCoroutine(LevelRoutine(level));
                    }
                }
                else
                {
                    ShowFinalLevelPrompt();

                    bio?.LogEvent("FinalLevelCompleted", "Final level completed successfully", currentLevel);
                    bio?.SetPhase("recovery_relaxation");

                    yield return new WaitForSeconds(8f);

                    if (finalLevelPrompt != null)
                        finalLevelPrompt.SetActive(false);

                    if (currentFollowingBanner == finalLevelPrompt)
                        currentFollowingBanner = null;

                    sessionComplete = true;

                    string chosenScene = PlayerPrefs.GetString("ChosenScene", relaxationSceneName);

                    if (!string.IsNullOrEmpty(chosenScene))
                    {
                        SceneManager.LoadScene(chosenScene);
                    }
                    else
                    {
                        Debug.LogWarning("No relaxation scene found.");
                    }
                }

                yield break;
            }

            yield return null;
        }
    }

    IEnumerator HandleElevation(int currentLevelIndex)
    {
        bio?.LogEvent(
            "RegressionTriggered",
            $"Regression from Level {currentLevelIndex} to Level {Mathf.Max(0, currentLevelIndex - 1)}",
            currentLevelIndex
        );

        companion?.NotifyRegression(Mathf.Max(0, currentLevelIndex - 1));

        yield return new WaitForSeconds(2f);

        int targetLevel = Mathf.Max(0, currentLevelIndex - 1);

        if (targetLevel == 0)
        {
            TeleportPlayer(level0Spawn);
            companion?.NotifyLevelEntered(0);
        }
        else
        {
            MoveToLevel(targetLevel);
        }

        panicHandling = false;

        bio?.SetLevel(targetLevel);
        bio?.SetPhase(targetLevel == 0 ? "prompt" : "exposure");
        bio?.SetSystemState("Calm");

        if (targetLevel == 0)
            StartCoroutine(Level0Routine());
        else
            StartCoroutine(LevelRoutine(targetLevel));
    }

    IEnumerator ReturnToRelaxation()
    {
        bio?.LogEvent("ReturnToRelaxation", "User returned to relaxation scene due to panic", currentLevel);
        bio?.SetPhase("recovery_relaxation");

        companion?.ClearQueue();
        companion?.NotifyPanicDetected();

        yield return new WaitForSeconds(2f);

        // Wait for NPC to finish its current line before transitioning
        if (companion != null)
        {
            float timeout = 8f; // safety cap so we never get stuck forever
            float waited = 0f;
            while (companion.GetRemainingAudioTime() > 0.1f && waited < timeout)
            {
                waited += Time.deltaTime;
                yield return null;
            }
        }

        string chosenScene = PlayerPrefs.GetString("ChosenScene", relaxationSceneName);

        if (!string.IsNullOrEmpty(chosenScene))
        {
            SceneManager.LoadScene(chosenScene);
        }
        else
        {
            Debug.LogWarning("No scene saved.");
            panicHandling = false;
            StartCoroutine(Level0Routine());
        }
    }

    void MoveToLevel(int level)
    {
        if (level == 0)
        {
            TeleportPlayer(level0Spawn);
            companion?.NotifyLevelEntered(0);
            return;
        }

        if (activePlayerSpawnPoints == null || level >= activePlayerSpawnPoints.Length)
        {
            Debug.LogWarning($"No player spawn configured for level {level}");
            return;
        }

        TeleportPlayer(activePlayerSpawnPoints[level]);

        if (companion != null)
        {
            companion.NotifyLevelEntered(level);

            if (activeNPCSpawnPoints != null &&
                level < activeNPCSpawnPoints.Length &&
                activeNPCSpawnPoints[level] != null)
            {
                companion.SetFixedPosition(activeNPCSpawnPoints[level]);
            }
            else
            {
                Debug.LogWarning($"No NPC spawn configured for level {level}");
            }
        }
    }

    void ShowProceedPrompt()
    {
        if (proceedPrompt == null) return;

        UpdateBannerPosition(proceedPrompt);
        proceedPrompt.SetActive(true);
        currentFollowingBanner = proceedPrompt;
    }

    void ShowFinalLevelPrompt()
    {
        if (finalLevelPrompt == null) return;

        UpdateBannerPosition(finalLevelPrompt);
        finalLevelPrompt.SetActive(true);
        currentFollowingBanner = finalLevelPrompt;

        companion?.NotifyCongrats();
    }

    void ShowPathChoicePrompt()
    {
        if (pathChoicePrompt == null) return;

        UpdateBannerPosition(pathChoicePrompt);
        pathChoicePrompt.SetActive(true);
        currentFollowingBanner = pathChoicePrompt;
    }

    void UpdateBannerPosition(GameObject banner)
    {
        if (banner == null) return;

        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;

        if (cameraTransform == null) return;

        banner.transform.position =
            cameraTransform.position +
            cameraTransform.forward * bannerDistanceFromCamera +
            cameraTransform.up * bannerHeightOffset;

        if (faceCamera)
            banner.transform.forward = cameraTransform.forward;
    }

    void TeleportPlayer(Transform spawnPoint)
    {
        if (player != null && spawnPoint != null)
        {
            player.position = spawnPoint.position;
            player.rotation = spawnPoint.rotation;
        }
    }

    public void OnProceedButtonPressed()
    {
        bio?.LogEvent("ProceedButtonPressed", $"Proceed button pressed at Level {currentLevel}", currentLevel);

        if (proceedPrompt != null)
            proceedPrompt.SetActive(false);

        if (currentFollowingBanner == proceedPrompt)
            currentFollowingBanner = null;

        if (currentLevel == 0)
        {
            ShowPathChoicePrompt();
        }
        else
        {
            userWantsToProceed = true;
            waitingForUser = false;
        }
    }

    public void OnStayButtonPressed()
    {
        bio?.LogEvent("StayButtonPressed", $"Stay button pressed at Level {currentLevel}", currentLevel);

        userWantsToProceed = false;

        if (proceedPrompt != null)
            proceedPrompt.SetActive(false);

        if (currentFollowingBanner == proceedPrompt)
            currentFollowingBanner = null;

        waitingForUser = false;
    }

    public void OnIndoorButtonPressed()
    {
        chosenPath = ExposurePath.Indoor;

        bio?.SetPath("Indoor");
        bio?.LogEvent("PathChosen", "Indoor path selected", 0, "Indoor");

        if (pathChoicePrompt != null)
            pathChoicePrompt.SetActive(false);

        if (currentFollowingBanner == pathChoicePrompt)
            currentFollowingBanner = null;

        userWantsToProceed = true;
        waitingForUser = false;
    }

    public void OnOutdoorButtonPressed()
    {
        chosenPath = ExposurePath.Outdoor;

        bio?.SetPath("Outdoor");
        bio?.LogEvent("PathChosen", "Outdoor path selected", 0, "Outdoor");

        if (pathChoicePrompt != null)
            pathChoicePrompt.SetActive(false);

        if (currentFollowingBanner == pathChoicePrompt)
            currentFollowingBanner = null;

        userWantsToProceed = true;
        waitingForUser = false;
    }
}