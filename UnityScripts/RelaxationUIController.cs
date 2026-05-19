using UnityEngine;

public class RelaxationUIController : MonoBehaviour
{
    void Start()
    {
        if (BaselineManager.Instance != null)
        {
            BaselineManager.Instance.StartBaseline();
        }
        else
        {
            Debug.LogWarning("BaselineManager not found. Running UI test mode.");
        }
    }

    public void ContinuePressed()
    {
        if (BaselineManager.Instance != null)
        {
            BaselineManager.Instance.StopBaseline();
        }
        else
        {
            Debug.LogWarning("BaselineManager not found. Skipping StopBaseline().");
        }

        if (GameManager.Instance != null)
        {
            GameManager.Instance.RelaxationFinished();
        }
        else
        {
            Debug.LogWarning("GameManager not found. Continue button was pressed.");
        }
    }
}