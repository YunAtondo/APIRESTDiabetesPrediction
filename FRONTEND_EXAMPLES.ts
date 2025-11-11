// ============================================
// SERVICIO ANGULAR PARA GESTIÓN DE MODELOS
// ============================================

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

// Interfaces
export interface ModelMetrics {
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
  loss: number;
}

export interface ModelVersionInfo {
  version: string;
  created_at: string;
  metrics: ModelMetrics;
  is_active: boolean;
  training_samples: number;
}

export interface RetrainingRequest {
  use_database: boolean;      // true = BD + CSV original combinados | false = solo CSV especificado
  dataset_name?: string;       // Requerido si use_database=false
  epochs?: number;             // Default: 190
  batch_size?: number;         // Default: 64
  learning_rate?: number;      // Default: 0.001
  hidden_size?: number;        // Default: 64
}

export interface RetrainingResponse {
  success: boolean;
  message: string;
  version: string;
  metrics: ModelMetrics;
  training_time: number;
}

export interface ModelListResponse {
  models: ModelVersionInfo[];
  active_model: string;
}

@Injectable({
  providedIn: 'root'
})
export class ModelManagementService {
  private apiUrl = 'http://localhost:8000/model';
  
  constructor(private http: HttpClient) {}
  
  private getHeaders(): HttpHeaders {
    const token = localStorage.getItem('access_token');
    return new HttpHeaders({
      'Authorization': `Bearer ${token}`
    });
  }
  
  // 1. Reentrenar modelo
  // use_database=true: Combina datos de BD + DiabetesDataset.csv original
  // use_database=false: Usa solo el CSV especificado en dataset_name
  retrainModel(request: RetrainingRequest): Observable<RetrainingResponse> {
    return this.http.post<RetrainingResponse>(
      `${this.apiUrl}/retrain`,
      request,
      { headers: this.getHeaders() }
    );
  }
  
  // 2. Listar versiones de modelos
  getModelVersions(): Observable<ModelListResponse> {
    return this.http.get<ModelListResponse>(
      `${this.apiUrl}/versions`,
      { headers: this.getHeaders() }
    );
  }
  
  // 3. Activar una versión
  activateModel(version: string): Observable<any> {
    return this.http.post(
      `${this.apiUrl}/activate`,
      { version },
      { headers: this.getHeaders() }
    );
  }
  
  // 4. Obtener modelo activo
  getActiveModel(): Observable<any> {
    return this.http.get(
      `${this.apiUrl}/active`,
      { headers: this.getHeaders() }
    );
  }
  
  // 5. Subir dataset
  uploadDataset(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const token = localStorage.getItem('access_token');
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${token}`
      // NO incluir Content-Type, Angular lo hace automáticamente
    });
    
    return this.http.post(
      `${this.apiUrl}/upload-dataset`,
      formData,
      { headers }
    );
  }
  
  // 6. Eliminar versión
  deleteModelVersion(version: string): Observable<any> {
    return this.http.delete(
      `${this.apiUrl}/version/${version}`,
      { headers: this.getHeaders() }
    );
  }
}


// ============================================
// COMPONENTE DE EJEMPLO
// ============================================

import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-model-management',
  templateUrl: './model-management.component.html'
})
export class ModelManagementComponent implements OnInit {
  models: ModelVersionInfo[] = [];
  activeModel: string = '';
  isLoading: boolean = false;
  isTraining: boolean = false;
  
  retrainingRequest: RetrainingRequest = {
    use_database: true,
    epochs: 190,
    batch_size: 64,
    learning_rate: 0.001,
    hidden_size: 64
  };
  
  constructor(private modelService: ModelManagementService) {}
  
  ngOnInit() {
    this.loadModels();
  }
  
  // Cargar lista de modelos
  loadModels() {
    this.isLoading = true;
    this.modelService.getModelVersions().subscribe({
      next: (response) => {
        this.models = response.models;
        this.activeModel = response.active_model;
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error al cargar modelos:', error);
        this.isLoading = false;
      }
    });
  }
  
  // Reentrenar modelo
  startTraining() {
    if (confirm('¿Está seguro de que desea reentrenar el modelo? Este proceso puede tardar varios minutos.')) {
      this.isTraining = true;
      
      this.modelService.retrainModel(this.retrainingRequest).subscribe({
        next: (response) => {
          alert(`Modelo entrenado exitosamente!\n` +
                `Versión: ${response.version}\n` +
                `Accuracy: ${(response.metrics.accuracy * 100).toFixed(2)}%\n` +
                `F1-Score: ${response.metrics.f1_score.toFixed(4)}\n` +
                `Tiempo: ${response.training_time.toFixed(2)}s`);
          
          this.isTraining = false;
          this.loadModels(); // Recargar lista
        },
        error: (error) => {
          alert('Error al entrenar el modelo: ' + error.error.detail);
          this.isTraining = false;
        }
      });
    }
  }
  
  // Activar modelo
  activateModel(version: string) {
    const model = this.models.find(m => m.version === version);
    if (!model) return;
    
    const message = `¿Desea activar este modelo?\n\n` +
                   `Versión: ${version}\n` +
                   `Accuracy: ${(model.metrics.accuracy * 100).toFixed(2)}%\n` +
                   `F1-Score: ${model.metrics.f1_score.toFixed(4)}`;
    
    if (confirm(message)) {
      this.modelService.activateModel(version).subscribe({
        next: (response) => {
          alert('Modelo activado exitosamente');
          this.loadModels();
        },
        error: (error) => {
          alert('Error al activar modelo: ' + error.error.detail);
        }
      });
    }
  }
  
  // Eliminar modelo
  deleteModel(version: string) {
    if (confirm(`¿Está seguro de eliminar la versión ${version}?`)) {
      this.modelService.deleteModelVersion(version).subscribe({
        next: (response) => {
          alert('Modelo eliminado exitosamente');
          this.loadModels();
        },
        error: (error) => {
          alert('Error al eliminar modelo: ' + error.error.detail);
        }
      });
    }
  }
  
  // Subir dataset
  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    
    if (file) {
      if (!file.name.endsWith('.csv')) {
        alert('Por favor, seleccione un archivo CSV');
        return;
      }
      
      this.isLoading = true;
      this.modelService.uploadDataset(file).subscribe({
        next: (response) => {
          alert(`Dataset subido exitosamente!\n` +
                `Archivo: ${response.filename}\n` +
                `Filas: ${response.rows}`);
          this.isLoading = false;
        },
        error: (error) => {
          alert('Error al subir dataset: ' + error.error.detail);
          this.isLoading = false;
        }
      });
    }
  }
  
  // Formatear porcentaje
  formatPercentage(value: number): string {
    return (value * 100).toFixed(2) + '%';
  }
  
  // Formatear fecha
  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleString();
  }
}


// ============================================
// TEMPLATE HTML DE EJEMPLO
// ============================================

/*
<div class="model-management-container">
  <h2>📊 Gestión de Modelos de ML</h2>
  
  <!-- Modelo Activo -->
  <div class="active-model-card" *ngIf="!isLoading && models.length > 0">
    <h3>🟢 Modelo Activo</h3>
    <div *ngFor="let model of models">
      <div *ngIf="model.is_active" class="model-info">
        <p><strong>Versión:</strong> {{ model.version }}</p>
        <p><strong>Accuracy:</strong> {{ formatPercentage(model.metrics.accuracy) }}</p>
        <p><strong>F1-Score:</strong> {{ model.metrics.f1_score.toFixed(4) }}</p>
        <p><strong>Precision:</strong> {{ formatPercentage(model.metrics.precision) }}</p>
        <p><strong>Recall:</strong> {{ formatPercentage(model.metrics.recall) }}</p>
        <p><strong>Creado:</strong> {{ formatDate(model.created_at) }}</p>
        <p><strong>Muestras:</strong> {{ model.training_samples }}</p>
      </div>
    </div>
  </div>
  
  <!-- Acciones -->
  <div class="actions">
    <button 
      (click)="startTraining()" 
      [disabled]="isTraining"
      class="btn btn-primary">
      <span *ngIf="!isTraining">🔄 Reentrenar Modelo</span>
      <span *ngIf="isTraining">⏳ Entrenando...</span>
    </button>
    
    <label class="btn btn-secondary">
      📤 Subir Dataset
      <input 
        type="file" 
        accept=".csv" 
        (change)="onFileSelected($event)" 
        style="display: none;">
    </label>
  </div>
  
  <!-- Configuración de Reentrenamiento -->
  <div class="training-config">
    <h4>⚙️ Configuración de Entrenamiento</h4>
    
    <div class="form-group">
      <label>
        <input 
          type="radio" 
          [(ngModel)]="retrainingRequest.use_database" 
          [value]="true">
        Usar Base de Datos + CSV Original (Recomendado - Más datos)
      </label>
      
      <label>
        <input 
          type="radio" 
          [(ngModel)]="retrainingRequest.use_database" 
          [value]="false">
        Usar solo Dataset CSV personalizado
      </label>
    </div>
    
    <div *ngIf="!retrainingRequest.use_database" class="form-group">
      <label>Nombre del Dataset:</label>
      <input 
        type="text" 
        [(ngModel)]="retrainingRequest.dataset_name" 
        placeholder="ejemplo.csv">
    </div>
    
    <div class="form-group">
      <label>Épocas:</label>
      <input 
        type="number" 
        [(ngModel)]="retrainingRequest.epochs" 
        min="1" 
        max="500">
    </div>
    
    <div class="form-group">
      <label>Batch Size:</label>
      <input 
        type="number" 
        [(ngModel)]="retrainingRequest.batch_size" 
        min="1" 
        max="256">
    </div>
  </div>
  
  <!-- Lista de Versiones -->
  <div class="versions-list">
    <h3>📋 Versiones Disponibles</h3>
    
    <div *ngIf="isLoading" class="loading">
      Cargando modelos...
    </div>
    
    <table *ngIf="!isLoading && models.length > 0">
      <thead>
        <tr>
          <th>Versión</th>
          <th>Accuracy</th>
          <th>F1-Score</th>
          <th>Muestras</th>
          <th>Fecha</th>
          <th>Estado</th>
          <th>Acciones</th>
        </tr>
      </thead>
      <tbody>
        <tr *ngFor="let model of models" [class.active]="model.is_active">
          <td>{{ model.version }}</td>
          <td>{{ formatPercentage(model.metrics.accuracy) }}</td>
          <td>{{ model.metrics.f1_score.toFixed(4) }}</td>
          <td>{{ model.training_samples }}</td>
          <td>{{ formatDate(model.created_at) }}</td>
          <td>
            <span *ngIf="model.is_active" class="badge badge-success">✅ Activo</span>
            <span *ngIf="!model.is_active" class="badge badge-secondary">⭕ Inactivo</span>
          </td>
          <td>
            <button 
              *ngIf="!model.is_active" 
              (click)="activateModel(model.version)"
              class="btn btn-sm btn-success">
              Activar
            </button>
            <button 
              *ngIf="!model.is_active" 
              (click)="deleteModel(model.version)"
              class="btn btn-sm btn-danger">
              Eliminar
            </button>
          </td>
        </tr>
      </tbody>
    </table>
    
    <p *ngIf="!isLoading && models.length === 0">
      No hay modelos disponibles. Entrena tu primer modelo.
    </p>
  </div>
</div>
*/


// ============================================
// EJEMPLO CON FETCH (JavaScript Vanilla)
// ============================================

// Obtener token del localStorage
const token = localStorage.getItem('access_token');

// Headers comunes
const headers = {
  'Authorization': `Bearer ${token}`,
  'Content-Type': 'application/json'
};

// 1. Reentrenar modelo con datos de BD + CSV original (COMBINADOS)
async function retrainFromDatabase() {
  try {
    const response = await fetch('http://localhost:8000/model/retrain', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        use_database: true,  // ← Combina BD + DiabetesDataset.csv original
        epochs: 190,
        batch_size: 64,
        learning_rate: 0.001,
        hidden_size: 64
      })
    });
    
    const data = await response.json();
    
    if (data.success) {
      console.log('Modelo entrenado:', data);
      console.log('Versión:', data.version);
      console.log('Accuracy:', (data.metrics.accuracy * 100).toFixed(2) + '%');
      console.log('F1-Score:', data.metrics.f1_score.toFixed(4));
      console.log('Tiempo:', data.training_time.toFixed(2) + 's');
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

// 2. Listar modelos
async function listModels() {
  try {
    const response = await fetch('http://localhost:8000/model/versions', {
      method: 'GET',
      headers: headers
    });
    
    const data = await response.json();
    console.log('Modelos disponibles:', data.models);
    console.log('Modelo activo:', data.active_model);
    
    return data;
  } catch (error) {
    console.error('Error:', error);
  }
}

// 3. Activar modelo
async function activateModel(version) {
  try {
    const response = await fetch('http://localhost:8000/model/activate', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({ version })
    });
    
    const data = await response.json();
    console.log('Modelo activado:', data);
  } catch (error) {
    console.error('Error:', error);
  }
}

// 4. Subir dataset
async function uploadDataset(fileInput) {
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch('http://localhost:8000/model/upload-dataset', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
        // NO incluir Content-Type para multipart/form-data
      },
      body: formData
    });
    
    const data = await response.json();
    console.log('Dataset subido:', data);
  } catch (error) {
    console.error('Error:', error);
  }
}
