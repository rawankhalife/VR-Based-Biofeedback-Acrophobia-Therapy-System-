using UnityEngine;

public class HouseVisibilityByCollider : MonoBehaviour
{
    public BoxCollider houseCollider;   // Interior collider
    public GameObject exterior;         // Exterior building to hide/show

    Transform cam;

    void Start()
    {
        cam = Camera.main.transform;

        if (houseCollider == null)
            houseCollider = GetComponent<BoxCollider>();

        if (houseCollider == null)
        {
            Debug.LogError("No BoxCollider found!");
            enabled = false;
        }
    }

    void Update()
    {
        Bounds b = houseCollider.bounds;   // World-space bounds
        Vector3 camPos = cam.position;

        bool inside =
            camPos.x > b.min.x && camPos.x < b.max.x &&
            camPos.y > b.min.y && camPos.y < b.max.y &&
            camPos.z > b.min.z && camPos.z < b.max.z;

        exterior.SetActive(!inside);
    }
}
