using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;

public class XRHoverGlow : MonoBehaviour
{
    Material mat;
    bool isHovering = false;
    Vector3 startPos;
    public float hoverHeight = 0.05f;   // how much it moves up
    public float moveSpeed = 10f;        // smoothness
    private XRSimpleInteractable interactable;

    void Start()
    {
        mat = GetComponent<Renderer>().material;
        startPos = transform.position;
        mat.EnableKeyword("_EMISSION");
        mat.SetColor("_EmissionColor", Color.black);

        // Get or add the XRSimpleInteractable component
        interactable = GetComponent<XRSimpleInteractable>();
        if (interactable == null)
        {
            interactable = gameObject.AddComponent<XRSimpleInteractable>();
        }

        // Subscribe to hover events
        interactable.hoverEntered.AddListener(OnHoverEnter);
        interactable.hoverExited.AddListener(OnHoverExit);
    }

    void OnDestroy()
    {
        // Unsubscribe to prevent memory leaks
        if (interactable != null)
        {
            interactable.hoverEntered.RemoveListener(OnHoverEnter);
            interactable.hoverExited.RemoveListener(OnHoverExit);
        }
    }

    void OnHoverEnter(HoverEnterEventArgs args)
    {
        mat.SetColor("_EmissionColor", Color.gray * 0.4f);
        isHovering = true;
    }

    void OnHoverExit(HoverExitEventArgs args)
    {
        mat.SetColor("_EmissionColor", Color.black);
        isHovering = false;
    }

    void Update()
    {
        // 🔹 MOVE CARD (smooth)
        Vector3 targetPos = isHovering
            ? startPos + Vector3.up * hoverHeight
            : startPos;

        transform.position = Vector3.Lerp(
            transform.position,
            targetPos,
            Time.deltaTime * moveSpeed
        );
    }
}