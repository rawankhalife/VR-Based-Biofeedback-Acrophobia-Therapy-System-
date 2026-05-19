using System.Collections;
using UnityEngine;
using UnityEngine.Video;
using UnityEngine.UI;

public class RelaxationVideoController : MonoBehaviour
{
    [Header("Popup Positioning")]
    public Transform cameraTransform;
    public float popupDistance = 1f;
    public float popupHeightOffset = -0.1f;
    public bool faceCamera = true;

    private bool followPopup = false;

    [Header("Video")]
    public VideoPlayer videoPlayer;

    [Header("2-Min Popup (World Space Canvas on Camera)")]
    public GameObject popupCanvas;       // The world-space canvas parented to camera

    [Header("Ground Gaze Button")]
    public GazeButton gazeButton;        // The GazeButton component on the ground button
    public GraphicRaycaster gazeRaycaster; // GraphicRaycaster on the ground button's canvas

    [Header("Fading")]
    public FadingScript fadingScript;

    [Header("Timing")]
    public float popupDelay = 120f;      // 2 minutes
    public float popupDuration = 5f;     // Auto-hide after 5 seconds

    private bool popupShown = false;
    private bool levelLoading = false;

    void Start()
    {
        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;

        // Hide popup at start
        if (popupCanvas != null)
            popupCanvas.SetActive(false);

        // Disable gaze button interaction at start
        SetGazeButtonActive(false);

        // Start baseline collection
        if (BaselineManager.Instance != null)
            BaselineManager.Instance.StartBaseline();

        // Hook into video end event
        if (videoPlayer != null)
            videoPlayer.loopPointReached += OnVideoFinished;

        // Start the 2-minute timer
        StartCoroutine(PopupTimer());
    }

    void Update()
    {
        if (followPopup)
            UpdatePopupPosition();
    }

    // ─── 2-Minute Timer ───────────────────────────────────────────
    private IEnumerator PopupTimer()
    {
        yield return new WaitForSeconds(popupDelay);

        if (!levelLoading)
            ShowPopup();
    }

    // ─── Show Popup ───────────────────────────────────────────────
    private void ShowPopup()
    {
        if (popupShown) return;
        popupShown = true;

        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;

        UpdatePopupPosition();

        if (popupCanvas != null)
            popupCanvas.SetActive(true);

        followPopup = true;

        SetGazeButtonActive(true);

        StartCoroutine(HidePopupAfterDelay());
    }

    private void UpdatePopupPosition()
    {
        if (popupCanvas == null || cameraTransform == null) return;

        popupCanvas.transform.position =
            cameraTransform.position +
            cameraTransform.forward * popupDistance +
            Vector3.up * popupHeightOffset;

        if (faceCamera)
            popupCanvas.transform.forward = cameraTransform.forward;
    }

    private IEnumerator HidePopupAfterDelay()
    {
        yield return new WaitForSeconds(popupDuration);

        followPopup = false;

        if (popupCanvas != null)
            popupCanvas.SetActive(false);
    }

    // ─── Video Finished Naturally ─────────────────────────────────
    private void OnVideoFinished(VideoPlayer vp)
    {
        if (levelLoading) return;
        levelLoading = true;

        if (BaselineManager.Instance != null)
            BaselineManager.Instance.StopBaseline();

        if (fadingScript != null)
            fadingScript.FadeOutAndLoadScene("Level0");
        else
            GameManager.Instance.LoadNextScene();
    }

    // ─── Called by GazeButton when activated ─────────────────────
    // (GazeButton already calls fadingScript.FadeOutAndLoadScene("Level0") directly)
    // But we still need to stop baseline when that happens.
    // Hook this into GazeButton.Activate() by adding an event, OR
    // just call StopBaseline here via a public method GazeButton can call.
    public void OnGazeCompleted()
    {
        if (levelLoading) return;
        levelLoading = true;

        if (BaselineManager.Instance != null)
            BaselineManager.Instance.StopBaseline();
    }

    // ─── Helpers ──────────────────────────────────────────────────
    private void SetGazeButtonActive(bool active)
    {
        if (gazeButton != null)
            gazeButton.enabled = active;

        if (gazeRaycaster != null)
            gazeRaycaster.enabled = active;

        if (gazeButton != null)
            gazeButton.gameObject.SetActive(active);
    }
}