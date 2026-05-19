using UnityEngine;

public class GazeRaycaster : MonoBehaviour
{
    private GazeButton currentGaze;

    void Update()
    {
        Ray ray = new Ray(transform.position, transform.forward);
        RaycastHit hit;

        // Use SphereCast for forgiveness
        if (Physics.SphereCast(ray, 0.05f, out hit))
        {
            // Look for GazeButton on hit object or its parents
            GazeButton gazeBtn = hit.collider.GetComponentInParent<GazeButton>();

            if (gazeBtn != null)
            {
                // Still looking at the SAME button → do nothing
                if (currentGaze == gazeBtn)
                    return;

                // Looking at a NEW button
                ClearGaze();
                currentGaze = gazeBtn;
                currentGaze.StartGaze();
                return;
            }
        }

        // Only clear gaze if we are NOT hitting a GazeButton
        ClearGaze();
    }

    void ClearGaze()
    {
        if (currentGaze != null)
        {
            currentGaze.StopGaze();
            currentGaze = null;
        }
    }
}
