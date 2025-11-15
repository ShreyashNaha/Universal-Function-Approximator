package ifndef.nn.util;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.List;

public class GraphingUtil {

    public static void plot(List<Double> xValues, List<Double> expected, List<Double> predicted, String title) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame(title);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Close only this window
            frame.setSize(800, 600);
            frame.setLocationRelativeTo(null); // Center on screen

            GraphPanel graphPanel = new GraphPanel(xValues, expected, predicted);
            frame.add(graphPanel);

            frame.setVisible(true);
        });
    }
}

class GraphPanel extends JPanel {

    private final List<Double> xValues;
    private final List<Double> expected;
    private final List<Double> predicted;

    private double panX = 0.0;
    private double panY = 0.0;
    private double scale = 20.0;
    private Point lastMousePos;

    public GraphPanel(List<Double> xValues, List<Double> expected, List<Double> predicted) {
        this.xValues = xValues;
        this.expected = expected;
        this.predicted = predicted;
        this.setBackground(Color.WHITE); 

        // --- Mouse Listener for Panning ---
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                lastMousePos = e.getPoint();
            }
        });

        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (lastMousePos != null) {
                    double dx = e.getX() - lastMousePos.x;
                    double dy = e.getY() - lastMousePos.y;

                    panX += dx;
                    panY += dy;

                    lastMousePos = e.getPoint();
                    repaint(); 
                }
            }
        });

        addMouseWheelListener((MouseWheelEvent e) -> {
            double zoomFactor = (e.getWheelRotation() < 0) ? 1.1 : 0.9; // 10% zoom

            Point mousePos = e.getPoint();
            double mouseWorldX = (mousePos.x - getWidth() / 2.0 - panX) / scale;
            double mouseWorldY = (mousePos.y - getHeight() / 2.0 - panY) / -scale;

            scale *= zoomFactor;

            double newMouseWorldX = (mousePos.x - getWidth() / 2.0 - panX) / scale;
            double newMouseWorldY = (mousePos.y - getHeight() / 2.0 - panY) / -scale;

            panX += (newMouseWorldX - mouseWorldX) * scale;
            panY -= (newMouseWorldY - mouseWorldY) * scale;

            repaint();
        });
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2d.translate(getWidth() / 2.0, getHeight() / 2.0);
        g2d.translate(panX, panY);
        g2d.scale(scale, -scale);

        drawAxes(g2d);

        drawPath(g2d, xValues, expected, Color.GREEN.darker());
        drawPath(g2d, xValues, predicted, Color.RED);

        drawLegend(g);
    }

    private void drawAxes(Graphics2D g2d) {
        double xMin = (-getWidth() / 2.0 - panX) / scale;
        double xMax = (getWidth() / 2.0 - panX) / scale;
        double yMin = (-getHeight() / 2.0 - panY) / -scale;
        double yMax = (getHeight() / 2.0 - panY) / -scale;
        
        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(1.0f / (float) scale)); // Thin lines
        
        g2d.drawLine((int) Math.floor(xMin), 0, (int) Math.ceil(xMax), 0);
        g2d.drawLine(0, (int) Math.floor(yMin), 0, (int) Math.ceil(yMax));
    }

    private void drawPath(Graphics2D g2d, List<Double> xs, List<Double> ys, Color color) {
        g2d.setColor(color);
        g2d.setStroke(new BasicStroke(2.0f / (float) scale)); // Thicker data line

        Path2D.Double path = new Path2D.Double();
        if (xs.isEmpty() || ys.isEmpty()) return;
        
        path.moveTo(xs.get(0), ys.get(0));
        for (int i = 1; i < xs.size(); i++) {
            path.lineTo(xs.get(i), ys.get(i));
        }
        g2d.draw(path);
    }

    private void drawLegend(Graphics g) {
        ((Graphics2D) g).setTransform(new java.awt.geom.AffineTransform());
        
        g.setColor(Color.GREEN.darker());
        g.drawString("Expected", 10, 20);
        g.setColor(Color.RED);
        g.drawString("Predicted", 10, 35);
    }
}