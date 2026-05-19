using UnityEngine;

public class MiddleZoneTracker : MonoBehaviour
{
    public float avoidanceTimeThreshold = 20f;
    public float repeatTalkInterval = 12f;
    public CompanionNPC npc;
    public SceneProgressionManager progressionManager;

    private int safeZoneContacts = 0;
    private float timeInsideSafeZone = 0f;
    private float talkTimer = 0f;
    private bool avoidanceDetected = false;

    void Update()
    {
        if (safeZoneContacts > 0)
        {
            timeInsideSafeZone += Time.deltaTime;

            if (!avoidanceDetected && timeInsideSafeZone >= avoidanceTimeThreshold)
            {
                avoidanceDetected = true;
                talkTimer = 0f;

                Debug.Log("Player stayed in middle zone too long");

                if (npc != null)
                    npc.NotifyAvoidingStimulus();
            }
            else if (avoidanceDetected)
            {
                talkTimer += Time.deltaTime;

                if (talkTimer >= repeatTalkInterval)
                {
                    talkTimer = 0f;

                    Debug.Log("Player still avoiding stimulus");

                    if (npc != null)
                        npc.NotifyAvoidingStimulus();
                }
            }
        }
        else
        {
            timeInsideSafeZone = 0f;
            talkTimer = 0f;
            avoidanceDetected = false;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("SafeZone"))
        {
            safeZoneContacts++;
            Debug.Log("Entered SafeZone: " + other.name + " | count = " + safeZoneContacts);

            if (progressionManager != null)
                progressionManager.SetSafeZone(safeZoneContacts > 0);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("SafeZone"))
        {
            safeZoneContacts = Mathf.Max(0, safeZoneContacts - 1);
            Debug.Log("Exited SafeZone: " + other.name + " | count = " + safeZoneContacts);

            if (progressionManager != null)
                progressionManager.SetSafeZone(safeZoneContacts > 0);
        }
    }
}