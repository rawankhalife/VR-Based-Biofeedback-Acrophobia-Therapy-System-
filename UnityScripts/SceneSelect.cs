using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;

public class XRSceneLoader : MonoBehaviour
{
    public string sceneName;
    private XRSimpleInteractable interactable;

    void Awake()
    {
        // Get or add the XRSimpleInteractable component
        interactable = GetComponent<XRSimpleInteractable>();
        if (interactable == null)
        {
            interactable = gameObject.AddComponent<XRSimpleInteractable>();
        }

        // Subscribe to the select event
        interactable.selectEntered.AddListener(OnSelect);
    }

    void OnDestroy()
    {
        // Unsubscribe to prevent memory leaks
        if (interactable != null)
        {
            interactable.selectEntered.RemoveListener(OnSelect);
        }
    }

    void OnSelect(SelectEnterEventArgs args)
    {
        Debug.Log("Object selected with VR controller");

        // Save the chosen scene
        PlayerPrefs.SetString("ChosenScene", sceneName);
        PlayerPrefs.Save();

        // Load the scene
        SceneManager.LoadScene(sceneName);
    }
}