using UnityEngine;
using UnityEngine.UI;

public class GazeButton : MonoBehaviour
{
    public float gazeTime = 5f;
    public Color normalColor = Color.white;
    public Color fillColor = new Color(0.5f, 0.5f, 0.5f, 0.3f); // Grey with low intensity (30% alpha)

    // NEW: Public reference to FadingScript
    public FadingScript fadingScript;

    // NEW: Reference to fill image
    public Image fillImage; // The image that fills from left to right

    private float timer = 0f;
    private bool isGazing = false;
    private Image buttonImage;

    void Start()
    {
        buttonImage = GetComponentInChildren<Image>();
        if (buttonImage == null)
        {
            Debug.LogError("No Image component found on ContinueButton or its children!");
        }
        else
        {
            buttonImage.color = normalColor;
        }

        // Setup fill image
        if (fillImage != null)
        {
            fillImage.fillAmount = 0f; // Start empty
            fillImage.type = Image.Type.Filled;
            fillImage.fillMethod = Image.FillMethod.Horizontal;
            fillImage.fillOrigin = (int)Image.OriginHorizontal.Left;
            fillImage.color = fillColor;
        }

        // Try to find fading script if not assigned
        if (fadingScript == null)
        {
            fadingScript = Object.FindObjectOfType(typeof(FadingScript)) as FadingScript;
        }
    }

    void Update()
    {
        if (isGazing)
        {
            timer += Time.deltaTime;

            // Update fill amount based on timer
            if (fillImage != null)
            {
                fillImage.fillAmount = Mathf.Clamp01(timer / gazeTime);
            }

            if (timer >= gazeTime)
            {
                Activate();
            }
        }
    }

    public void StartGaze()
    {
        isGazing = true;
        timer = 0f;

        if (fillImage != null)
        {
            fillImage.fillAmount = 0f; // Reset fill
        }
    }

    public void StopGaze()
    {
        isGazing = false;
        timer = 0f;

        if (fillImage != null)
        {
            fillImage.fillAmount = 0f; // Reset fill
        }
    }

    void Activate()
    {
        isGazing = false;

        if (fillImage != null)
        {
            fillImage.fillAmount = 0f; // Reset fill
        }

        Debug.Log("GAZE ACTIVATED - Fading to Level0");
        RelaxationVideoController controller = Object.FindObjectOfType<RelaxationVideoController>();
        if (controller != null)
            controller.OnGazeCompleted();

        if (fadingScript != null)
        {
            fadingScript.FadeOutAndLoadScene("Level0");
        }
        else
        {
            Debug.LogWarning("No FadingScript found! Loading scene directly.");
            GameManager.Instance.RelaxationFinished();
        }
    }
}