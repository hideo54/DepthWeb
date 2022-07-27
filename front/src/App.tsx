import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { IconSpan } from '@hideo54/reactor';
import { CloudUpload } from '@styled-icons/ionicons-outline';
import './App.css';

const App = () => {
  const apiEndpoint = '/depth-web/api/predict';
  const bucketName = 'depth-web';
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [depthImageUrl, setDepthImageUrl] = useState<string | null>(null);
  const [depth, setDepth] = useState<number[][] | null>(null);

  const onFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    setErrorMessage(null);
    if (files && files.length > 0) {
      const file = files[0];
      const formData = new FormData();
      formData.append('file', file);
      setIsLoading(true);
      setDepthImageUrl(null);
      setDepth(null);
      const res = await fetch(apiEndpoint, {
        method: 'POST',
        body: formData,
      });
      if (res.status === 200) {
        const { depthPoints, filename } = await res.json();
        setDepthImageUrl(`https://storage.googleapis.com/${bucketName}/${filename}`);
        setDepth(depthPoints);
      } else {
        setErrorMessage('エラーが発生しました');
      }
      setIsLoading(false);
    } else {
      setErrorMessage('適切なファイルが選択されていません');
    }
  };
  return (
    <div className="App">
      <h1>DepthWeb</h1>
      {isLoading ? (
        <p>Loading...</p>
      ) : (
        <label className='upload'>
          <IconSpan LeftIcon={CloudUpload}>画像をアップロード</IconSpan>
          <input type='file' name='file' accept='image/png, image/jpeg' required onChange={onFileUpload} />
        </label>
      )}
      {errorMessage && <p>{errorMessage}</p>}
      {depthImageUrl && depth &&
        <section>
          <h2>深度画像</h2>
          <img src={depthImageUrl} alt='Depth' />
          <h2>点群</h2>
          <Plot
            data={[{
              x: depth.map(d => d[0]),
              y: depth.map(d => d[1]),
              z: depth.map(d =>
                d[2] * Math.min(
                  depth[depth.length - 1][0],
                  depth[depth.length - 1][1]
                )
              ),
              type: 'scatter3d',
              mode: 'markers',
              marker: {
                size: 0.6,
                color: depth.map(d => d[3]),
              },
            }]}
            layout={{
              scene: {
                aspectmode: 'data',
                camera: {
                  eye: { x: -0.5, y: -0.5, z: 1.5 },
                  up: { x: -100, y: -1, z: 0 },
                },
              },
            }}
          />
        </section>
      }
    </div>
  );
};

export default App;
