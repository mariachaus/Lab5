import React, { useState } from 'react';
import { View, Button, Text, ScrollView, StyleSheet, TouchableOpacity } from 'react-native';
import { Audio } from 'expo-av';

export default function AudioRecorder() {
  const [recording, setRecording] = useState(null);
  const [statusText, setStatusText] = useState('');
  const [predictions, setPredictions] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [lastRecordingUri, setLastRecordingUri] = useState(null);
  const [sound, setSound] = useState(null);

  async function startRecording() {
    try {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        setStatusText('Permission denied');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const rec = new Audio.Recording();
      await rec.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
      await rec.startAsync();
      setRecording(rec);
      setStatusText('Recording...');
      setPredictions({});
    } catch (error) {
      console.error('Failed to start recording', error);
      setStatusText('Failed to start recording');
    }
  }

  async function stopRecording() {
    if (!recording) return;
    
    setIsLoading(true);
    setStatusText('Processing...');

    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      setLastRecordingUri(uri); 

      const formData = new FormData();
      formData.append('file', {
        uri: uri,
        type: 'audio/m4a',
        name: `recording-${Date.now()}.m4a`,
      });

      console.log('Uploading audio...');

      const response = await fetch('http://10.132.230.98:8000/predict/', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await response.json();
      console.log('Server response:', result);

      if (result.error) {
        setStatusText(`Error: ${result.error}`);
      } else {
        setPredictions(result.predictions || {});
        setStatusText('Analysis complete!');
        
        const topPrediction = Object.entries(result.predictions || {})
          .sort(([,a], [,b]) => b - a)[0];
        
        if (topPrediction) {
          setStatusText(`Detected: ${topPrediction[0]} (${topPrediction[1]}%)`);
        }
      }

    } catch (error) {
      console.error('Upload failed:', error);
      setStatusText(`Upload failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }

  async function playRecording() {
    if (!lastRecordingUri) {
      setStatusText('No recording to play');
      return;
    }

    try {
      setStatusText('Playing recording...');
      
      if (sound) {
        await sound.stopAsync();
        await sound.unloadAsync();
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      const { sound: newSound } = await Audio.Sound.createAsync(
        { uri: lastRecordingUri },
        { shouldPlay: true }
      );

      setSound(newSound);

      newSound.setOnPlaybackStatusUpdate((status) => {
        if (status.didJustFinish) {
          setStatusText('Playback finished');
        }
      });

      await newSound.playAsync();
      
    } catch (error) {
      console.error('Playback failed', error);
      setStatusText('Playback failed');
    }
  }

  async function stopPlayback() {
    if (sound) {
      try {
        await sound.stopAsync();
        await sound.unloadAsync();
        setSound(null);
        setStatusText('Playback stopped');
      } catch (error) {
        console.error('Stop playback failed', error);
      }
    }
  }

  function clearAll() {
    if (recording) {
      recording.stopAndUnloadAsync();
      setRecording(null);
    }

    if (sound) {
      sound.stopAsync();
      sound.unloadAsync();
      setSound(null);
    }

    setLastRecordingUri(null);
    setPredictions({});
    setStatusText('Cleared');
  }

  React.useEffect(() => {
    return sound
      ? () => {
          console.log('Unloading Sound');
          sound.unloadAsync();
        }
      : undefined;
  }, [sound]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Animal Sound Detector</Text>
      
      {/* Основні кнопки запису */}
      <View style={styles.buttonContainer}>
        <Button 
          title="Start Recording" 
          onPress={startRecording}
          disabled={!!recording || isLoading}
        />
        <Button 
          title="Stop & Analyze" 
          onPress={stopRecording}
          disabled={!recording || isLoading}
          color="#FF4444"
        />
      </View>

      {/* Кнопки управління записом */}
      {(lastRecordingUri || predictions.length > 0) && (
        <View style={styles.controlContainer}>
          <Text style={styles.controlTitle}>Recording Controls:</Text>
          <View style={styles.controlButtons}>
            <TouchableOpacity 
              style={[styles.controlButton, !lastRecordingUri && styles.disabledButton]}
              onPress={playRecording}
              disabled={!lastRecordingUri}
            >
              <Text style={styles.controlButtonText}>▶️ Play</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.controlButton, !sound && styles.disabledButton]}
              onPress={stopPlayback}
              disabled={!sound}
            >
              <Text style={styles.controlButtonText}>⏹️ Stop</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.controlButton, styles.clearButton]}
              onPress={clearAll}
            >
              <Text style={styles.controlButtonText}>🗑️ Clear</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      <Text style={styles.status}>{statusText}</Text>

      {isLoading && (
        <Text style={styles.loading}>Processing audio...</Text>
      )}

      {Object.keys(predictions).length > 0 && (
        <View style={styles.results}>
          <Text style={styles.resultsTitle}>Detection Results:</Text>
          <ScrollView style={styles.resultsScroll}>
            {Object.entries(predictions)
              .sort(([,a], [,b]) => b - a)
              .map(([animal, percentage]) => (
                <View key={animal} style={styles.resultItem}>
                  <Text style={styles.animalName}>{animal}</Text>
                  <Text style={[
                    styles.percentage,
                    percentage > 50 && styles.highPercentage,
                    percentage > 30 && percentage <= 50 && styles.mediumPercentage
                  ]}>
                    {percentage}%
                  </Text>
                </View>
              ))}
          </ScrollView>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginTop: 50,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 30,
    color: '#333',
  },
  buttonContainer: {
    gap: 10,
    marginBottom: 20,
  },
  controlContainer: {
    marginBottom: 20,
    padding: 15,
    backgroundColor: 'white',
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  controlTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  controlButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  controlButton: {
    flex: 1,
    padding: 12,
    marginHorizontal: 5,
    backgroundColor: '#007AFF',
    borderRadius: 8,
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#ccc',
  },
  clearButton: {
    backgroundColor: '#FF3B30',
  },
  controlButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 14,
  },
  status: {
    textAlign: 'center',
    fontSize: 16,
    marginBottom: 20,
    color: '#666',
    padding: 10,
    backgroundColor: 'white',
    borderRadius: 8,
  },
  loading: {
    textAlign: 'center',
    fontSize: 16,
    color: 'blue',
    marginBottom: 20,
    fontStyle: 'italic',
  },
  results: {
    flex: 1,
    marginTop: 10,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  resultsScroll: {
    flex: 1,
  },
  resultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
    marginVertical: 4,
    backgroundColor: 'white',
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  animalName: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  percentage: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#666',
  },
  highPercentage: {
    color: '#4CAF50', 
  },
  mediumPercentage: {
    color: '#FF9800', 
  },
});