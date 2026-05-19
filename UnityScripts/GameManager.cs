using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
    public static GameManager Instance;
    public int currentLevel = 0;

    private void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);
        Debug.Log("GameManager created");
    }

    // Called when relaxation finishes (now just sets level, scene load handled by FadingScript)
    public void RelaxationFinished()
    {
        Debug.Log("Relaxation finished → Level0 will load after fade");
        currentLevel = 0;
    }

    public void LoadNextScene()
    {
        if (currentLevel == 0)
        {
            SceneManager.LoadScene("Level0");
        }
        else
        {
            SceneManager.LoadScene("Level" + currentLevel);
        }
    }

    public void LevelFinished()
    {
        currentLevel++;
        SceneManager.LoadScene("Level" + currentLevel);
    }
}