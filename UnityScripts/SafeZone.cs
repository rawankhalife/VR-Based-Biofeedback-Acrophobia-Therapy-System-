using UnityEngine;

public class SafeZoneTrigger : MonoBehaviour
{
    public SceneProgressionManager manager;

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("MainCamera"))
        {
            manager.SetSafeZone(true);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("MainCamera"))
        {
            manager.SetSafeZone(false);
        }
    }
}