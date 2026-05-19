using System.Collections;
using UnityEngine;

public class ZoneAudioManager : MonoBehaviour
{
    [Header("Zone Audio Clips")]
    public AudioClip natureClip;
    public AudioClip urbanClip;
    public AudioClip windClip;

    [Header("Audio Sources")]
    public AudioSource sourceA;
    public AudioSource sourceB;

    [Header("Settings")]
    public float fadeDuration = 1.5f;
    public float volume = 0.8f;

    private AudioSource _active;
    private AudioSource _inactive;
    private Coroutine _fadeCoroutine;

    void Start()
    {
        sourceA.loop = true;
        sourceB.loop = true;
        sourceA.playOnAwake = false;
        sourceB.playOnAwake = false;
        sourceA.volume = 0f;
        sourceB.volume = 0f;
        _active = sourceA;
        _inactive = sourceB;
    }

    public void PlayZone(AudioClip clip)
    {
        if (clip == null) return;
        if (_active.clip == clip && _active.isPlaying) return;

        (_active, _inactive) = (_inactive, _active);

        _active.clip = clip;
        _active.volume = 0f;
        _active.Play();

        if (_fadeCoroutine != null)
            StopCoroutine(_fadeCoroutine);

        _fadeCoroutine = StartCoroutine(Crossfade());
    }

    IEnumerator Crossfade()
    {
        float t = 0f;
        float startVol = _inactive.volume;

        while (t < fadeDuration)
        {
            t += Time.deltaTime;
            float ratio = t / fadeDuration;
            _active.volume = Mathf.Lerp(0f, volume, ratio);
            _inactive.volume = Mathf.Lerp(startVol, 0f, ratio);
            yield return null;
        }

        _active.volume = volume;
        _inactive.volume = 0f;
        _inactive.Stop();
    }
}