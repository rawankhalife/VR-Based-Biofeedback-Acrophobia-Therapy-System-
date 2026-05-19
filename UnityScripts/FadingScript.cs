using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

public class FadingScript : MonoBehaviour
{
    public CanvasGroup canvasGroup;
    public float fadeDuration = 1.0f;
    public bool autoFadeOnStart = false;
    public bool fadeIn = false;

    // NEW: Reference to the loading text
    public GameObject loadingPanel;
    public float blackScreenDuration = 5f; // How long to stay black

    private Coroutine currentFade;

    public void Start()
    {
        // Make sure loading panel starts hidden
        if (loadingPanel != null)
        {
            loadingPanel.SetActive(false);
        }

        if (autoFadeOnStart)
        {
            if (fadeIn)
            {
                FadeIn();
            }
            else
            {
                FadeOut();
            }
        }
    }

    public void FadeIn()
    {
        if (currentFade != null) StopCoroutine(currentFade);
        currentFade = StartCoroutine(FadeCanvasGroup(canvasGroup, canvasGroup.alpha, 0, fadeDuration));
    }

    public void FadeOut()
    {
        if (currentFade != null) StopCoroutine(currentFade);
        currentFade = StartCoroutine(FadeCanvasGroup(canvasGroup, canvasGroup.alpha, 1, fadeDuration));
    }

    public void FadeOutAndLoadScene(string sceneName)
    {
        if (currentFade != null) StopCoroutine(currentFade);
        currentFade = StartCoroutine(FadeOutThenLoad(sceneName));
    }

    private IEnumerator FadeOutThenLoad(string sceneName)
    {
        // Fade to black
        float startAlpha = canvasGroup.alpha;
        yield return StartCoroutine(FadeCanvasGroup(canvasGroup, startAlpha, 1, fadeDuration));

        // Show loading text
        if (loadingPanel != null)
        {
            loadingPanel.SetActive(true);
        }

        // Stay black with text visible
        yield return new WaitForSeconds(blackScreenDuration);

        // Load the scene
        SceneManager.LoadScene(sceneName);
    }

    private IEnumerator FadeCanvasGroup(CanvasGroup cg, float start, float end, float duration)
    {
        float elapsedTime = 0.0f;
        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            cg.alpha = Mathf.Lerp(start, end, elapsedTime / duration);
            yield return null;
        }
        cg.alpha = end;
    }
}