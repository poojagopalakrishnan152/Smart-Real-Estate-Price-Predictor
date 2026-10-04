// static/script.js
// Client-side interactions, validation, loading animations, and Chart.js analytics

document.addEventListener("DOMContentLoaded", function () {
  // 1. Mobile Navigation Toggle
  const mobileToggle = document.getElementById("mobileToggle");
  const navLinks = document.getElementById("navLinks");
  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener("click", function () {
      navLinks.classList.toggle("active");
    });
  }

  // 2. Alert Dismissal
  const alertCloseBtns = document.querySelectorAll(".alert-close");
  alertCloseBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      const alert = btn.closest(".alert");
      if (alert) {
        alert.style.opacity = "0";
        setTimeout(function () {
          alert.remove();
        }, 200);
      }
    });
  });

  // 3. Prediction Form Validation & Loading State
  const predictForm = document.getElementById("predictForm");
  const loadingOverlay = document.getElementById("loadingOverlay");

  if (predictForm) {
    predictForm.addEventListener("submit", function (e) {
      const locationInput = document.getElementById("locationSelect");
      const sqftInput = document.getElementById("totalSqft");

      // Validate Location
      if (!locationInput || !locationInput.value.trim()) {
        e.preventDefault();
        alert("Please select a Bengaluru location.");
        if (locationInput) locationInput.focus();
        return false;
      }

      // Validate Area
      const sqftVal = parseFloat(sqftInput.value);
      if (isNaN(sqftVal) || sqftVal <= 0) {
        e.preventDefault();
        alert("Please enter a valid property area in square feet (greater than 0).");
        if (sqftInput) sqftInput.focus();
        return false;
      }

      if (sqftVal < 100 || sqftVal > 50000) {
        e.preventDefault();
        alert("Please enter a property area between 100 and 50,000 sq.ft.");
        if (sqftInput) sqftInput.focus();
        return false;
      }

      // Show Loading Screen
      if (loadingOverlay) {
        loadingOverlay.style.display = "flex";
      }
    });
  }

  // 4. Quick Location Selection from Cloud/Tags
  const locationTags = document.querySelectorAll(".location-tag");
  const locationSelect = document.getElementById("locationSelect");
  if (locationTags.length > 0 && locationSelect) {
    locationTags.forEach(function (tag) {
      tag.addEventListener("click", function () {
        const selectedLoc = tag.getAttribute("data-location");
        if (selectedLoc) {
          locationSelect.value = selectedLoc;
          // Trigger change event if needed
          locationSelect.dispatchEvent(new Event("change"));
        }
      });
    });
  }

  // 5. Initialize Chart.js Dashboard if on Home Page
  initAnalyticsCharts();
});

/**
 * Fetches dataset analytics from /api/analytics and renders interactive Chart.js charts
 */
function initAnalyticsCharts() {
  const priceChartCanvas = document.getElementById("priceByLocationChart");
  const areaTypeChartCanvas = document.getElementById("areaTypeChart");
  const countChartCanvas = document.getElementById("countByLocationChart");

  if (!priceChartCanvas && !areaTypeChartCanvas && !countChartCanvas) {
    return;
  }

  fetch("/api/analytics")
    .then(function (response) {
      if (!response.ok) {
        throw new Error("Analytics API returned status " + response.status);
      }
      return response.json();
    })
    .then(function (data) {
      if (!data.success) return;

      // Chart 1: Top Bengaluru Locations by Average Price
      if (priceChartCanvas && data.top_by_price) {
        new Chart(priceChartCanvas, {
          type: "bar",
          data: {
            labels: data.top_by_price.locations,
            datasets: [
              {
                label: "Average Price (₹ Lakhs)",
                data: data.top_by_price.avg_prices,
                backgroundColor: "rgba(13, 148, 136, 0.75)",
                borderColor: "rgba(13, 148, 136, 1)",
                borderWidth: 1.5,
                borderRadius: 4
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: function (ctx) {
                    return "₹ " + ctx.raw + " Lakhs";
                  }
                }
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                title: {
                  display: true,
                  text: "Price in Lakhs (₹)",
                  font: { weight: "600" }
                }
              },
              x: {
                ticks: {
                  maxRotation: 45,
                  minRotation: 25,
                  font: { size: 11 }
                }
              }
            }
          }
        });
      }

      // Chart 2: Property Distribution by Area Type (Doughnut)
      if (areaTypeChartCanvas && data.area_types) {
        new Chart(areaTypeChartCanvas, {
          type: "doughnut",
          data: {
            labels: data.area_types.labels,
            datasets: [
              {
                data: data.area_types.counts,
                backgroundColor: [
                  "#0d9488",
                  "#3b82f6",
                  "#f59e0b",
                  "#8b5cf6"
                ],
                hoverOffset: 6
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: "bottom",
                labels: { boxWidth: 14, font: { size: 12 } }
              },
              tooltip: {
                callbacks: {
                  label: function (ctx) {
                    return ctx.label + ": " + ctx.raw.toLocaleString() + " properties";
                  }
                }
              }
            }
          }
        });
      }

      // Chart 3: Most Popular Localities by Listing Density
      if (countChartCanvas && data.top_by_count) {
        new Chart(countChartCanvas, {
          type: "bar",
          data: {
            labels: data.top_by_count.locations,
            datasets: [
              {
                label: "Property Listings Count",
                data: data.top_by_count.counts,
                backgroundColor: "rgba(59, 130, 246, 0.75)",
                borderColor: "rgba(59, 130, 246, 1)",
                borderWidth: 1.5,
                borderRadius: 4
              }
            ]
          },
          options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false }
            },
            scales: {
              x: {
                beginAtZero: true,
                title: {
                  display: true,
                  text: "Number of Properties in Dataset",
                  font: { weight: "600" }
                }
              }
            }
          }
        });
      }
    })
    .catch(function (err) {
      console.warn("Could not load analytics charts:", err);
    });
}
