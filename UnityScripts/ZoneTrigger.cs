using UnityEngine;

public class ZoneTrigger : MonoBehaviour
{
    public enum ZoneType
    {
        Nature,
        Urban,
        Wind
    }

    public ZoneType zoneType;
    public ZoneAudioManager audioManager;

    private void OnTriggerEnter(Collider other)
    {
        if (!other.GetComponentInParent<CharacterController>()) return;
        if (audioManager == null) return;

        switch (zoneType)
        {
            case ZoneType.Nature:
                audioManager.PlayZone(audioManager.natureClip);
                break;

            case ZoneType.Urban:
                audioManager.PlayZone(audioManager.urbanClip);
                break;

            case ZoneType.Wind:
                audioManager.PlayZone(audioManager.windClip);
                break;
        }
    }
}