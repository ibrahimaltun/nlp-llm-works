import React, { useState } from "react";
import {
  Box,
  Container,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Paper,
  Divider,
} from "@mui/material";
import {
  Send as SendIcon,
  MenuBook as BookIcon,
  SmartToy as RobotIcon,
  Image as ImageIcon,
  WarningAmber as WarningIcon,
} from "@mui/icons-material";

interface ChatResponse {
  response: string;
  source_page: number | null;
  shown_image: string | null;
}

export const OfflineHelpAssistant: React.FC = () => {
  const [query, setQuery] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const apiResponse = await fetch("http://127.0.0.1:8091", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query }),
      });

      if (!apiResponse.ok)
        throw new Error("Sunucu hatası veya bağlantı kesildi.");

      const data: ChatResponse = await apiResponse.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(
        "Sistemle iletişim kurulamadı. Python API sunucunuzun açık olduğundan emin olun.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container
      maxWidth="md"
      sx={{
        py: 4,
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        gap: 3,
      }}
    >
      {/* Header Panel */}
      <Paper
        elevation={0}
        variant="outlined"
        sx={{ p: 3, borderRadius: 3, bgcolor: "background.paper" }}
      >
        <Typography
          variant="h4"
          component="h1"
          fontWeight="bold"
          color="text.primary"
          gutterBottom
        >
          Çevrimdışı Akıllı Kılavuz Asistanı
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Yerel RAG mimarisi ve çok modlu (multimodal) yapay zeka motoru ile
          tamamen internetsiz çalışır.
        </Typography>
      </Paper>

      {/* Input Form Area */}
      <Box
        component="form"
        onSubmit={handleSearch}
        sx={{ display: "flex", gap: 1 }}
      >
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Kılavuzda merak ettiğiniz bir özelliği sorun... (Örn: Cihaz kurulumu nasıl yapılır?)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
          slotProps={{
            input: {
              sx: { borderRadius: 3, bgcolor: "background.paper" },
            },
          }}
        />
        <Button
          type="submit"
          variant="contained"
          color="primary"
          disabled={loading}
          endIcon={
            loading ? (
              <CircularProgress size={20} color="inherit" />
            ) : (
              <SendIcon />
            )
          }
          sx={{
            px: 4,
            borderRadius: 3,
            textTransform: "none",
            fontWeight: "bold",
          }}
        >
          {loading ? "İnceleniyor" : "Sor"}
        </Button>
      </Box>

      {/* Loading Spinner View */}
      {loading && (
        <Paper
          variant="outlined"
          sx={{
            p: 6,
            textAlign: "center",
            borderStyle: "dashed",
            borderRadius: 3,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 2,
          }}
        >
          <CircularProgress size={40} />
          <Typography
            variant="body1"
            fontWeight="medium"
            color="text.secondary"
          >
            Veritabanı taranıyor ve yapay zeka görseli analiz ediyor...
          </Typography>
        </Paper>
      )}

      {/* Error Alert Box */}
      {error && (
        <Paper
          variant="outlined"
          sx={{
            p: 3,
            borderColor: "error.light",
            bgcolor: "error.lighter",
            borderRadius: 3,
            display: "flex",
            alignItems: "center",
            gap: 2,
          }}
        >
          <WarningIcon color="error" />
          <Typography variant="body2" color="error.main" fontWeight="medium">
            {error}
          </Typography>
        </Paper>
      )}

      {/* Main AI Response Container */}
      {result && (
        <Card
          variant="outlined"
          sx={{ borderRadius: 3, boxShadow: "0 4px 20px rgba(0,0,0,0.05)" }}
        >
          <CardHeader
            avatar={<RobotIcon color="primary" />}
            title={
              <Typography
                variant="subtitle2"
                fontWeight="bold"
                sx={{
                  trackingLetter: "0.5px",
                  textTransform: "uppercase",
                  color: "text.secondary",
                }}
              >
                Yapay Zeka Yanıtı
              </Typography>
            }
            action={
              result.source_page && (
                <Chip
                  icon={<BookIcon fontSize="small" />}
                  label={`Kaynak: Sayfa ${result.source_page}`}
                  color="primary"
                  variant="soft"
                  sx={{ fontWeight: "bold" }}
                />
              )
            }
            sx={{
              borderBottom: "1px solid",
              borderColor: "divider",
              bgcolor: "grey.50",
            }}
          />
          <CardContent sx={{ p: 4 }}>
            {/* AI Generated Text Response */}
            <Typography
              variant="body1"
              color="text.primary"
              sx={{
                lineHeight: 1.7,
                whiteSpace: "pre-wrap",
                mb: result.shown_image ? 4 : 0,
              }}
            >
              {result.response}
            </Typography>

            {/* Inline Document Target Snapshot Mapping */}
            {result.shown_image && (
              <Box>
                <Divider sx={{ my: 3 }} />
                <Box
                  sx={{
                    maxWidth: 500,
                    mx: "auto",
                    p: 1.5,
                    border: "1px solid",
                    borderColor: "divider",
                    borderRadius: 3,
                    bgcolor: "grey.50",
                  }}
                >
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 1,
                      mb: 1.5,
                    }}
                  >
                    <ImageIcon color="action" fontSize="small" />
                    <Typography
                      variant="caption"
                      fontWeight="medium"
                      color="text.secondary"
                    >
                      İncelenen Kılavuz Görseli / Ekran Görüntüsü:
                    </Typography>
                  </Box>
                  <Box
                    component="img"
                    src={`data:image/jpeg;base64,${result.shown_image}`}
                    alt="Document Reference Snapshot"
                    sx={{
                      width: "100%",
                      height: "auto",
                      maxHeight: 320,
                      objectFit: "contain",
                      borderRadius: 2,
                      display: "block",
                      bgcolor: "white",
                    }}
                  />
                </Box>
              </Box>
            )}
          </CardContent>
        </Card>
      )}
    </Container>
  );
};
