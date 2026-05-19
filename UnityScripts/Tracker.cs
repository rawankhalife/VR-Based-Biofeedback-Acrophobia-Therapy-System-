using UnityEngine;

public class Tracker : MonoBehaviour
{
    public float avoidanceTimeThreshold = 25f; // seconds before NPC reacts

    float timeAwayFromStimulus = 0f;
    int stimulusContacts = 0; // how many colliders player is inside

    public CompanionNPC npc; // drag your NPC here

    bool hasTriggered = false;

    void Update()
    {
        // If player is NOT inside any stimulus collider
        if (stimulusContacts == 0)
        {
            timeAwayFromStimulus += Time.deltaTime;

            if (timeAwayFromStimulus >= avoidanceTimeThreshold && !hasTriggered)
            {
                hasTriggered = true;

                // Trigger NPC
                Debug.Log("Player avoiding height stimulus");

                if (npc != null)
                {
                    npc.NotifyAvoidingStimulus(); // we'll add this
                }
            }
        }
        else
        {
            // Player is near an edge → reset
            timeAwayFromStimulus = 0f;
            hasTriggered = false;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("HeightStimulus"))
        {
            stimulusContacts++;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("HeightStimulus"))
        {
            stimulusContacts--;
        }
    }
}
